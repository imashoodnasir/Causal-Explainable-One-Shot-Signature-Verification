class GraphTransformerExplainable(nn.Module):
    def __init__(self, d_model=64, n_heads=4):
        super(GraphTransformerExplainable, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, batch_first=True)
        self.proj = nn.Linear(d_model, 2)  # Binary: Genuine vs Forged

    def forward(self, x):
        x_attn, weights = self.attn(x, x, x)
        out = self.proj(x_attn.mean(1))
        return out, weights
