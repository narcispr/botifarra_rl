import torch as T
import torch.nn as nn
import torch.nn.functional as F

class AttnPool(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(d, d),
            nn.Tanh(),
            nn.Linear(d, 1)
        )

    def forward(self, h, mask: T.Tensor = None):  # h: (B,48,d), mask opcional (B,48) {0,1}
        s = self.score(h)                          # (B,48,1)
        if mask is not None:
            # cartes no vàlides: -inf al logit perquè no contribueixin
            s = s.masked_fill(mask.unsqueeze(-1) == 0, -T.finfo(h.dtype).max)
        a = T.softmax(s, dim=1)                   # (B,48,1)
        g = (a * h).sum(dim=1, keepdim=True)      # (B,1,d)
        return g

class CardDQN(nn.Module):
    def __init__(self,
                 d: int = 128,
                 head_hidden: int = 128,
                 use_transformer: bool = True,
                 n_layers: int = 1,
                 n_heads: int = 4):
        super().__init__()

        self.per_pos = nn.Sequential(
            nn.Linear(5, d),
            nn.ReLU(),
        )

        self.card_embed = nn.Embedding(48, d)

        self.use_transformer = use_transformer
        if use_transformer:
            enc_layer = nn.TransformerEncoderLayer(
                d_model=d,
                nhead=n_heads,
                dim_feedforward=4*d,
                activation='relu',
                batch_first=True,
                dropout=0.1,          # <- una mica d’estabilitat
                norm_first=True       # <- pre-norm, sol anar més fi
            )
            self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
            self.pre_ln = nn.LayerNorm(d)         # <- opcional: LN abans del mixer

        self.pool = AttnPool(d)

        self.head = nn.Sequential(
            nn.Linear(2*d, head_hidden),
            nn.ReLU(),
            nn.Linear(head_hidden, 1)
        )

        device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(device)

        # (Opcional) inicialització una mica millor
        nn.init.xavier_uniform_(self.per_pos[0].weight)
        nn.init.zeros_(self.per_pos[0].bias)
        nn.init.xavier_uniform_(self.head[0].weight)
        nn.init.zeros_(self.head[0].bias)

    def _reshape_input(self, state: T.Tensor) -> T.Tensor:
        if state.dim() == 2:
            B, D = state.shape
            if D != 48 * 5:
                raise ValueError(f"Esperava 240 features aplanades, rebut {D}")
            return state.view(B, 48, 5)
        elif state.dim() == 3:
            if state.shape[1:] == (48, 5):
                return state
            if state.shape[1:] == (5, 48):
                return state.transpose(1, 2)
        raise ValueError(f"state ha de ser (B,48,5) o (B,240) o (B,5,48); rebut {tuple(state.shape)}")

    def forward(self, state: T.Tensor, legal_mask: T.Tensor = None) -> T.Tensor:
        """
        Retorna Q(s) per carta: (B,48).
        Si passes legal_mask (B,48) {0,1}, també s’utilitza per al context (no ponderar cartes il·legals).
        """
        x = self._reshape_input(state)
        x = x.to(x.device)                 # no canvia, però mantenim simetria
        B = x.size(0)

        # Per-carta
        h = self.per_pos(x)                # (B,48,d)

        # Identitat de carta
        card_ids = T.arange(48, device=x.device).unsqueeze(0).expand(B, -1)  # (B,48)
        h = h + self.card_embed(card_ids)  # (B,48,d)

        # Mixer (Transformer)
        if self.use_transformer:
            h = self.pre_ln(h)             # opcional però útil
            h = self.encoder(h)            # (B,48,d)

        # Context global (attn-pool) amb mask opcional
        g = self.pool(h, legal_mask)       # (B,1,d)

        # Fusió local + global
        z = T.cat([h, g.expand(-1, 48, -1)], dim=-1)  # (B,48,2d)

        # Q per carta
        q = self.head(z).squeeze(-1)       # (B,48)
        return self.apply_action_mask(q, legal_mask) if legal_mask is not None else q

    @staticmethod
    def apply_action_mask(q_values: T.Tensor, legal_mask: T.Tensor) -> T.Tensor:
        NEG_INF = -T.finfo(q_values.dtype).max
        return q_values.masked_fill(legal_mask == 0, NEG_INF)

    def update_weights(self, source_net: nn.Module, soft: bool = True, tau: float = 0.001) -> None:
        if soft:
            for tp, sp in zip(self.parameters(), source_net.parameters()):
                tp.data.copy_(tau * sp.data + (1.0 - tau) * sp.data)
        else:
            self.load_state_dict(source_net.state_dict())
