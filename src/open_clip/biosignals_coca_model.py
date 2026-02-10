"""
Biosignals-Text CoCa Model

Adapted from the original CoCa model to work with biosignals (time series) data
instead of images. This model is designed for biosignals-text contrastive learning.
"""

from typing import Dict, List, Optional, Union, Tuple
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import math
from dataclasses import dataclass, field

from .transformer import (
    LayerNormFp32,
    LayerNorm,
    QuickGELU,
    MultimodalTransformer,
    ConcatMultimodalTransformer,
)
from .model import CLIPTextCfg, _build_text_tower
from .coca_model import MultimodalCfg, _build_text_decoder_tower, _token_to_tensor

try:
    from transformers.generation.beam_search import BeamSearchScorer
    from transformers.generation.logits_process import (
        LogitsProcessorList,
        TopPLogitsWarper,
        TopKLogitsWarper,
        RepetitionPenaltyLogitsProcessor,
        MinLengthLogitsProcessor,
    )
    from transformers.generation.stopping_criteria import (
        MaxLengthCriteria,
        EosTokenCriteria,
        StoppingCriteriaList,
    )

    GENERATION_TYPES = {
        "top_k": TopKLogitsWarper,
        "top_p": TopPLogitsWarper,
        "beam_search": "beam_search"
    }
    _has_transformers = True
except ImportError as e:
    GENERATION_TYPES = {
        "top_k": None,
        "top_p": None,
        "beam_search": "beam_search"
    }
    _has_transformers = False


# ============================================================================
# Pure Transformer Architecture Components (from PureTransformerMAE)
# ============================================================================

class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE)"""
    def __init__(self, dim: int, theta: float = 10000.0, learned_freq: bool = False):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.learned_freq = learned_freq
        
        if learned_freq:
            # Learnable frequencies for channel attention
            self.freqs = nn.Parameter(torch.randn(dim // 2) * 0.02)
        else:
            # Fixed frequencies for temporal attention
            freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
            self.register_buffer('freqs', freqs)
    
    def rotate_queries_or_keys(self, x: torch.Tensor, position_ids: Optional[torch.Tensor] = None):
        """
        Apply rotary embeddings to queries or keys
        
        Args:
            x: (batch_size, num_heads, seq_len, head_dim)
            position_ids: (seq_len,) or (batch_size, seq_len) - position indices
        Returns:
            Rotated tensor of same shape
        """
        batch_size, num_heads, seq_len, head_dim = x.shape
        assert head_dim == self.dim, f"head_dim {head_dim} != self.dim {self.dim}"
        
        # Generate position indices if not provided
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=x.device, dtype=torch.float)
        elif position_ids.ndim == 2:
            # If 2D, take the first batch (assuming all batches have same pattern)
            position_ids = position_ids[0].float()
        else:
            position_ids = position_ids.float()
        
        # Compute angles: position_ids * freqs
        # position_ids: (seq_len,), freqs: (dim // 2,)
        # angles: (seq_len, dim // 2)
        angles = torch.einsum('s,d->sd', position_ids, self.freqs)
        
        # Duplicate for cos and sin
        # cos/sin: (seq_len, dim)
        cos = torch.cos(angles).repeat_interleave(2, dim=-1)
        sin = torch.sin(angles).repeat_interleave(2, dim=-1)
        
        # Reshape for broadcasting: (1, 1, seq_len, dim)
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)
        
        # Apply rotation
        # Split x into even and odd dimensions
        x1 = x[..., 0::2]  # Even dimensions
        x2 = x[..., 1::2]  # Odd dimensions
        
        # Apply rotation: [x1, x2] @ [[cos, -sin], [sin, cos]]
        x_rotated = torch.empty_like(x)
        x_rotated[..., 0::2] = x1 * cos[..., 0::2] - x2 * sin[..., 0::2]
        x_rotated[..., 1::2] = x1 * sin[..., 1::2] + x2 * cos[..., 1::2]
        
        return x_rotated


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization"""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class SwiGLU(nn.Module):
    """SwiGLU activation function: SiLU(x * W1) * (x * W2)"""
    def __init__(self, dim_in: int, dim_out: int, bias: bool = False):
        super().__init__()
        self.w1 = nn.Linear(dim_in, dim_out, bias=bias)
        self.w2 = nn.Linear(dim_in, dim_out, bias=bias)
        
    def forward(self, x):
        return F.silu(self.w1(x)) * self.w2(x)


class MLP(nn.Module):
    """MLP with configurable activation and normalization"""
    def __init__(self, 
                 dim: int, 
                 hidden_dim: int, 
                 dropout: float = 0.0,
                 activation: str = "swiglu",  # "swiglu", "gelu", "relu"
                 bias: bool = False):
        super().__init__()
        self.activation = activation
        
        if activation == "swiglu":
            # SwiGLU requires different structure: two parallel linear layers
            self.gate_proj = SwiGLU(dim, hidden_dim, bias=bias)
            self.down_proj = nn.Linear(hidden_dim, dim, bias=bias)
        else:
            # Standard MLP structure
            self.up_proj = nn.Linear(dim, hidden_dim, bias=bias)
            self.down_proj = nn.Linear(hidden_dim, dim, bias=bias)
            
            if activation == "gelu":
                self.act_fn = nn.GELU()
            elif activation == "relu":
                self.act_fn = nn.ReLU()
            else:
                raise ValueError(f"Unknown activation: {activation}")
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        if self.activation == "swiglu":
            x = self.gate_proj(x)
            x = self.dropout(x)
            x = self.down_proj(x)
        else:
            x = self.up_proj(x)
            x = self.act_fn(x)
            x = self.dropout(x)
            x = self.down_proj(x)
            
        return self.dropout(x)


class ChannelPatching(nn.Module):
    """Patching layer that operates independently on each channel"""
    def __init__(self, 
                 patch_size: int = 32,
                 conv_embed_dim: int = 256,
                 num_channels: int = 21):
        super().__init__()
        self.patch_size = patch_size
        self.conv_embed_dim = conv_embed_dim
        self.num_channels = num_channels
        
        # Single conv layer applied to all channels (kernel_size=patch_size, stride=patch_size)
        self.conv_patching = nn.Conv1d(
            in_channels=1, 
            out_channels=conv_embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            padding=0  # No padding for clean non-overlapping patches
        )
        
    def forward(self, x):
        """
        Args:
            x: (batch_size, num_channels, signal_length) - multi-channel signal
        Returns:
            (batch_size, num_channels, num_patches, conv_embed_dim) - patched representations
        """
        batch_size, num_channels, seq_len = x.shape
        
        # Reshape to process all channels independently: (batch_size * num_channels, 1, seq_len)
        x_reshaped = x.reshape(batch_size * num_channels, 1, seq_len)
        
        # Apply conv patching to all channels
        patched = self.conv_patching(x_reshaped)  # (batch_size * num_channels, conv_embed_dim, num_patches)
        
        # Reshape back to separate batch and channel dimensions
        _, conv_embed_dim, num_patches = patched.shape
        patched = patched.reshape(batch_size, num_channels, conv_embed_dim, num_patches)
        
        # Transpose to get (batch_size, num_channels, num_patches, conv_embed_dim)
        patched = patched.transpose(2, 3)
        
        return patched


class DualRoPEAttention(nn.Module):
    """Multi-head attention with separate RoPE for temporal and learnable RoPE for channels"""
    def __init__(self,
                 embed_dim: int = 256,
                 num_heads: int = 8,
                 dropout: float = 0.1,
                 attention_type: str = "temporal",  # "temporal" or "channel"
                 num_channels: int = 21,
                 shared_channel_rope: Optional[nn.Module] = None):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.attention_type = attention_type
        
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        # Linear projections
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        # RoPE embeddings - different for temporal vs channel
        if attention_type == "temporal":
            # Standard RoPE for temporal attention
            self.rotary_emb = RotaryEmbedding(
                dim=self.head_dim,
                theta=10000,
                learned_freq=False
            )
        elif attention_type == "channel":
            # Use shared learnable RoPE for channel attention if provided
            if shared_channel_rope is not None:
                self.rotary_emb = shared_channel_rope
            else:
                # Fallback to creating own RoPE
                self.rotary_emb = RotaryEmbedding(
                    dim=self.head_dim,
                    theta=10000,
                    learned_freq=True  # Learnable frequencies for channels
                )
        else:
            raise ValueError(f"Unknown attention_type: {attention_type}")
        
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5
        
    def forward(self, x, position_ids=None):
        """
        Args:
            x: (batch_size, seq_len, embed_dim)
            position_ids: (batch_size, seq_len) or (seq_len,) - custom position indices for RoPE
        Returns:
            (batch_size, seq_len, embed_dim)
        """
        batch_size, seq_len, embed_dim = x.shape
        
        # Linear projections
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Apply RoPE
        q = self.rotary_emb.rotate_queries_or_keys(q, position_ids=position_ids)
        k = self.rotary_emb.rotate_queries_or_keys(k, position_ids=position_ids)
        
        # Scaled dot-product attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)
        
        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        output = self.out_proj(attn_output)
        
        return output


class DualTransformerBlock(nn.Module):
    """Biosignal transformer block with channel and temporal attention using dual RoPE"""
    def __init__(self,
                 embed_dim: int = 256,
                 num_heads: int = 8,
                 num_temporal_layers: int = 2,
                 dropout: float = 0.1,
                 mlp_ratio: float = 4.0,
                 num_channels: int = 21,
                 activation: str = "swiglu",
                 norm_type: str = "rmsnorm",
                 mlp_bias: bool = False,
                 shared_channel_rope: Optional[nn.Module] = None):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_temporal_layers = num_temporal_layers
        
        # Helper function to create normalization layer
        def create_norm(dim):
            if norm_type == "rmsnorm":
                return RMSNorm(dim)
            elif norm_type == "layernorm":
                return nn.LayerNorm(dim)
            else:
                raise ValueError(f"Unknown norm_type: {norm_type}")
        
        # Channel-wise attention with shared learnable RoPE
        self.channel_attention = DualRoPEAttention(
            embed_dim, num_heads, dropout, 
            attention_type="channel", num_channels=num_channels,
            shared_channel_rope=shared_channel_rope
        )
        self.channel_norm = create_norm(embed_dim)
        
        # Temporal attention layers with standard RoPE
        self.temporal_attention_layers = nn.ModuleList([
            DualRoPEAttention(embed_dim, num_heads, dropout, attention_type="temporal") 
            for _ in range(num_temporal_layers)
        ])
        self.temporal_norms = nn.ModuleList([
            create_norm(embed_dim) 
            for _ in range(num_temporal_layers)
        ])
        
        # MLP layers
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.channel_mlp = MLP(
            dim=embed_dim,
            hidden_dim=mlp_hidden_dim,
            dropout=dropout,
            activation=activation,
            bias=mlp_bias
        )
        
        self.temporal_mlps = nn.ModuleList([
            MLP(
                dim=embed_dim,
                hidden_dim=mlp_hidden_dim,
                dropout=dropout,
                activation=activation,
                bias=mlp_bias
            ) for _ in range(num_temporal_layers)
        ])
        
        self.channel_mlp_norm = create_norm(embed_dim)
        self.temporal_mlp_norms = nn.ModuleList([
            create_norm(embed_dim) 
            for _ in range(num_temporal_layers)
        ])
        
    def forward(self, x, temporal_position_ids=None):
        """
        Args:
            x: (batch_size, num_channels, num_patches, embed_dim)
            temporal_position_ids: (batch_size, num_patches) or (num_patches,) - position indices for temporal RoPE
        Returns:
            (batch_size, num_channels, num_patches, embed_dim)
        """
        batch_size, num_channels, num_patches, embed_dim = x.shape
        
        # 1. Channel-wise attention on each patch independently
        x_for_channel_attn = x.permute(0, 2, 1, 3).contiguous().reshape(batch_size * num_patches, num_channels, embed_dim)
        
        # Apply channel attention with learnable RoPE
        channel_attn_out = self.channel_attention(x_for_channel_attn)
        
        # Residual connection and layer norm
        x_for_channel_attn = self.channel_norm(x_for_channel_attn + channel_attn_out)
        
        # MLP
        channel_mlp_out = self.channel_mlp(x_for_channel_attn)
        x_for_channel_attn = self.channel_mlp_norm(x_for_channel_attn + channel_mlp_out)
        
        # Reshape back
        x = x_for_channel_attn.reshape(batch_size, num_patches, num_channels, embed_dim).permute(0, 2, 1, 3)
        
        # 2. Temporal attention on patches for each channel
        x_for_temporal_attn = x.reshape(batch_size * num_channels, num_patches, embed_dim)
        
        # Prepare temporal position IDs
        if temporal_position_ids is not None:
            if temporal_position_ids.ndim == 2:
                temporal_pos_ids_expanded = temporal_position_ids[0]
            else:
                temporal_pos_ids_expanded = temporal_position_ids
        else:
            temporal_pos_ids_expanded = None
        
        # Apply multiple temporal attention layers
        for i in range(self.num_temporal_layers):
            temporal_attn_out = self.temporal_attention_layers[i](x_for_temporal_attn, position_ids=temporal_pos_ids_expanded)
            x_for_temporal_attn = self.temporal_norms[i](x_for_temporal_attn + temporal_attn_out)
            
            temporal_mlp_out = self.temporal_mlps[i](x_for_temporal_attn)
            x_for_temporal_attn = self.temporal_mlp_norms[i](x_for_temporal_attn + temporal_mlp_out)
        
        # Reshape back
        x = x_for_temporal_attn.reshape(batch_size, num_channels, num_patches, embed_dim)
        
        return x


# ============================================================================
# End of Pure Transformer Architecture Components
# ============================================================================


def _build_signal_tower(
        embed_dim: int,
        signal_cfg,
        output_tokens: bool = False,
        cast_dtype: Optional[torch.dtype] = None,
):
    """Build a biosignals encoder tower
    
    Args:
        embed_dim: Output embedding dimension
        signal_cfg: BiosignalsCfg or dict with configuration
        output_tokens: Whether to output tokens for multimodal decoder
        cast_dtype: Optional dtype for casting
    
    Returns:
        Biosignals encoder (either BiosignalsEncoder or PureTransformerBiosignalsEncoder)
    """
    if isinstance(signal_cfg, dict):
        signal_cfg = BiosignalsCfg(**signal_cfg)
    
    import logging
    architecture = getattr(signal_cfg, 'architecture', 'conv_transformer')
    logging.info(f"Building biosignals encoder with architecture: {architecture}")
    
    if architecture == "pure_transformer":
        signal_encoder = PureTransformerBiosignalsEncoder(
            biosignals_cfg=signal_cfg,
            embed_dim=embed_dim,
            output_tokens=output_tokens,
            cast_dtype=cast_dtype
        )
        logging.info(f"Pure Transformer architecture:")
        logging.info(f"  Patch size: {signal_cfg.patch_size}")
        logging.info(f"  Conv embed dim: {signal_cfg.conv_embed_dim}")
        logging.info(f"  Transformer blocks: {signal_cfg.transformer_layers}")
        logging.info(f"  Temporal layers per block: {signal_cfg.num_temporal_layers}")
        logging.info(f"  Activation: {signal_cfg.activation}")
        logging.info(f"  Norm type: {signal_cfg.norm_type}")
        logging.info(f"  Share channel RoPE: {signal_cfg.share_channel_rope}")
    elif architecture == "conv_transformer":
        signal_encoder = BiosignalsEncoder(
            biosignals_cfg=signal_cfg,
            embed_dim=embed_dim,
            output_tokens=output_tokens,
            cast_dtype=cast_dtype
        )
        logging.info(f"Conv-Transformer architecture:")
        logging.info(f"  Conv layers: {signal_cfg.conv_layers}")
        logging.info(f"  Kernel sizes: {signal_cfg.kernel_sizes}")
        logging.info(f"  Strides: {signal_cfg.strides}")
        logging.info(f"  Transformer layers: {signal_cfg.transformer_layers}")
    else:
        raise ValueError(f"Unknown architecture: {architecture}. Must be 'conv_transformer' or 'pure_transformer'")
    
    return signal_encoder


def _build_text_decoder_tower_v2(
        embed_dim,
        multimodal_cfg,
        quick_gelu: bool = False,
        cast_dtype: Optional[torch.dtype] = None,
        decoder_type: str = "cross_attention",
        prefix_len: int = 0,
):
    """Build text decoder tower with support for different decoder types.
    
    Args:
        embed_dim: Embedding dimension
        multimodal_cfg: MultimodalCfg config
        quick_gelu: Whether to use QuickGELU
        cast_dtype: Optional dtype for casting
        decoder_type: "cross_attention" or "concat"
            - "cross_attention": Uses separate cross-attention layers (default CoCa)
            - "concat": Concatenates image/biosignals and text tokens
        prefix_len: Number of prefix tokens (condition embeddings) prepended to text
            Used to pre-build prefix-causal attention mask
    """
    multimodal_cfg = MultimodalCfg(**multimodal_cfg) if isinstance(multimodal_cfg, dict) else multimodal_cfg
    act_layer = QuickGELU if quick_gelu else nn.GELU
    norm_layer = (
        LayerNormFp32 if cast_dtype in (torch.float16, torch.bfloat16) else LayerNorm
    )

    if decoder_type == "cross_attention":
        decoder = MultimodalTransformer(
            context_length=multimodal_cfg.context_length,
            width=multimodal_cfg.width,
            heads=multimodal_cfg.heads,
            layers=multimodal_cfg.layers,
            mlp_ratio=multimodal_cfg.mlp_ratio,
            ls_init_value=multimodal_cfg.ls_init_value,
            output_dim=embed_dim,
            act_layer=act_layer,
            norm_layer=norm_layer,
            prefix_len=prefix_len,
        )
    elif decoder_type == "concat":
        decoder = ConcatMultimodalTransformer(
            context_length=multimodal_cfg.context_length,
            width=multimodal_cfg.width,
            heads=multimodal_cfg.heads,
            layers=multimodal_cfg.layers,
            mlp_ratio=multimodal_cfg.mlp_ratio,
            ls_init_value=multimodal_cfg.ls_init_value,
            output_dim=embed_dim,
            act_layer=act_layer,
            norm_layer=norm_layer,
            prefix_len=prefix_len,
        )
    else:
        raise ValueError(f"Unknown decoder_type: {decoder_type}. Must be 'cross_attention' or 'concat'")

    return decoder


@dataclass
class BiosignalsCfg:
    """Configuration for biosignals encoder"""
    input_channels: int = 12  # Number of input channels (e.g., 12-lead ECG)
    signal_length: int = 1000  # Length of input time series
    sampling_rate: int = 500  # Sampling rate in Hz
    
    # Architecture selection
    architecture: str = "conv_transformer"  # "conv_transformer" or "pure_transformer"
    
    # Architecture parameters for conv_transformer
    conv_layers: List[int] = None  # Conv layer dimensions
    kernel_sizes: List[int] = None  # Kernel sizes for conv layers
    strides: List[int] = None  # Strides for conv layers
    
    # Architecture parameters for pure_transformer
    patch_size: int = 32  # Patch size for pure_transformer
    conv_embed_dim: int = 256  # Conv embedding dimension for pure_transformer
    num_temporal_layers: int = 2  # Number of temporal attention layers per block
    activation: str = "swiglu"  # "swiglu", "gelu", "relu" (for pure_transformer)
    norm_type: str = "rmsnorm"  # "rmsnorm", "layernorm" (for pure_transformer)
    mlp_bias: bool = False  # Whether to use bias in MLP layers (for pure_transformer)
    share_channel_rope: bool = True  # Share channel RoPE across blocks (for pure_transformer)
    decoder_tokens: int = 32  # Number of decoder tokens for dual-axis transformer (pure_transformer)
    
    # Transformer parameters (shared)
    transformer_layers: int = 6  # Number of transformer layers/blocks
    transformer_width: int = 768  # Transformer width
    transformer_heads: int = 12  # Number of attention heads
    mlp_ratio: float = 4.0  # MLP expansion ratio
    
    # Pooling and output
    pool_type: str = 'attn'  # 'avg', 'max', 'cls', 'attn'
    dropout: float = 0.1
    
    def __post_init__(self):
        if self.architecture == "conv_transformer":
            if self.conv_layers is None:
                # Default conv layers for processing time series
                self.conv_layers = [64, 128, 256, 512]
            if self.kernel_sizes is None:
                # Default kernel sizes
                self.kernel_sizes = [7, 5, 3, 3]
            if self.strides is None:
                # Default strides
                self.strides = [2, 2, 2, 2]


class BaseBiosignalsEncoder(nn.Module):
    """
    Base class for biosignals encoders that handles common pooling and projection logic.
    Child classes should implement _encode() to return features before pooling.
    """
    
    def __init__(
        self,
        biosignals_cfg: BiosignalsCfg,
        embed_dim: int,
        output_tokens: bool,
        transformer_width: int,
        cast_dtype: Optional[torch.dtype] = None
    ):
        super().__init__()
        self.biosignals_cfg = biosignals_cfg
        self.embed_dim = embed_dim
        self.output_tokens = output_tokens
        self.transformer_width = transformer_width
        self.pool_type = biosignals_cfg.pool_type
        
        # Projection to output embedding dimension
        self.proj_to_embed = nn.Linear(transformer_width, embed_dim)
        
        # Attention pooling if needed
        if self.pool_type == 'attn':
            self.attn_pool = nn.MultiheadAttention(
                transformer_width,
                biosignals_cfg.transformer_heads,
                batch_first=True
            )
        
    def _pool_features(self, x: torch.Tensor, has_cls_token: bool) -> torch.Tensor:
        """
        Pool features using the configured pooling method.
        
        Args:
            x: Features of shape (batch_size, seq_len, width)
            has_cls_token: Whether the sequence includes a CLS token at the last position
            
        Returns:
            pooled: Pooled features of shape (batch_size, width)
        """
        if self.pool_type == 'cls':
            # Use class token (last position)
            pooled = x[:, -1]
        elif self.pool_type == 'avg':
            # Average pooling over sequence
            if has_cls_token:
                pooled = x[:, :-1].mean(dim=1)
            else:
                pooled = x.mean(dim=1)
        elif self.pool_type == 'max':
            # Max pooling over sequence
            if has_cls_token:
                pooled = x[:, :-1].max(dim=1)[0]
            else:
                pooled = x.max(dim=1)[0]
        elif self.pool_type == 'attn':
            # Attention pooling using cls token as query
            query = x[:, -1:]  # CLS token as query
            # CLS attends to content tokens
            pooled, _ = self.attn_pool(query, x[:, :-1], x[:, :-1])
            pooled = pooled.squeeze(1)
        else:
            raise ValueError(f"Unknown pool_type: {self.pool_type}")
        
        return pooled
    
    def _encode(self, biosignals: torch.Tensor) -> Tuple[torch.Tensor, bool]:
        """
        Encode biosignals to features. Must be implemented by child classes.
        
        Args:
            biosignals: Input biosignals tensor
            
        Returns:
            features: Encoded features of shape (batch_size, seq_len, transformer_width)
            has_cls_token: Whether the sequence includes a CLS token at the last position
        """
        raise NotImplementedError("Child classes must implement _encode()")
    
    def forward(self, biosignals: torch.Tensor):
        """
        Forward pass with encoding, pooling, and projection.
        
        Args:
            biosignals: Input biosignals tensor
            
        Returns:
            embedding: Global embedding (batch_size, embed_dim)
            tokens_for_decoder: Optional tokens for decoder (batch_size, seq_len, transformer_width)
        """
        # Encode to features
        features, has_cls_token = self._encode(biosignals)
        
        # Pool features
        pooled = self._pool_features(features, has_cls_token)
        
        # Project to final embedding dimension
        embedding = self.proj_to_embed(pooled)
        
        if self.output_tokens:
            # Return tokens for multimodal decoder
            if has_cls_token:
                # Exclude CLS token from tokens for decoder
                tokens_for_decoder = features[:, :-1]
            else:
                tokens_for_decoder = features
            return embedding, tokens_for_decoder
        else:
            return embedding
    
    def set_grad_checkpointing(self, enable=True):
        # For compatibility with other models
        pass


class Conv1dBlock(nn.Module):
    """1D Convolutional block with normalization and activation"""
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, 
                 norm_layer=nn.BatchNorm1d, act_layer=nn.ReLU):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size, 
            stride=stride, padding=kernel_size//2
        )
        self.norm = norm_layer(out_channels)
        self.act = act_layer()
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.dropout(x)
        return x


class BiosignalsEncoder(BaseBiosignalsEncoder):
    """
    Biosignals encoder that converts time series data to embeddings.
    Uses a combination of 1D convolutions and transformers.
    """
    
    def __init__(
        self, 
        biosignals_cfg: BiosignalsCfg,
        embed_dim: int = 512,
        output_tokens: bool = False,
        cast_dtype: Optional[torch.dtype] = None
    ):
        # Initialize base class with common pooling/projection logic
        super().__init__(
            biosignals_cfg=biosignals_cfg,
            embed_dim=embed_dim,
            output_tokens=output_tokens,
            transformer_width=biosignals_cfg.transformer_width,
            cast_dtype=cast_dtype
        )
        
        # Convolutional feature extraction
        conv_layers = []
        in_channels = biosignals_cfg.input_channels
        
        for i, (out_channels, kernel_size, stride) in enumerate(
            zip(biosignals_cfg.conv_layers, biosignals_cfg.kernel_sizes, biosignals_cfg.strides)
        ):
            conv_layers.append(
                Conv1dBlock(in_channels, out_channels, kernel_size, stride)
            )
            in_channels = out_channels
            
        self.conv_layers = nn.Sequential(*conv_layers)
        
        # Calculate the length after convolutions with padding - we'll use a dummy forward pass
        # to get the exact dimensions
        with torch.no_grad():
            dummy_input = torch.randn(1, biosignals_cfg.input_channels, biosignals_cfg.signal_length)
            dummy_output = self.conv_layers(dummy_input)
            conv_output_length = dummy_output.shape[2]
        
        self.conv_output_length = conv_output_length
        self.conv_output_dim = biosignals_cfg.conv_layers[-1]
        
        # Projection to transformer dimension
        self.proj_conv_to_transformer = nn.Linear(
            self.conv_output_dim, biosignals_cfg.transformer_width
        )
        
        # Positional embeddings for sequence positions (excluding CLS token)
        # CLS token gets no positional embedding as it represents global context
        self.pos_embed = nn.Parameter(
            torch.randn(1, conv_output_length, biosignals_cfg.transformer_width)
        )
        
        # Add a class token for global representation (only used for 'cls' and 'attn' pooling)
        self.cls_token = nn.Parameter(
            torch.randn(1, 1, biosignals_cfg.transformer_width)
        )
        
        # Transformer layers
        norm_layer = LayerNormFp32 if cast_dtype in (torch.float16, torch.bfloat16) else LayerNorm
        act_layer = QuickGELU
        
        self.transformer_layers = nn.ModuleList([
            TransformerBlock(
                biosignals_cfg.transformer_width,
                biosignals_cfg.transformer_heads,
                biosignals_cfg.mlp_ratio,
                act_layer=act_layer,
                norm_layer=norm_layer,
                dropout=biosignals_cfg.dropout
            )
            for _ in range(biosignals_cfg.transformer_layers)
        ])
        
        # Final layer norm
        self.ln_final = norm_layer(biosignals_cfg.transformer_width)
            
    def _encode(self, biosignals):
        """
        Encode biosignals to features before pooling.
        
        Args:
            biosignals: Tensor of shape (batch_size, channels, signal_length)
        Returns:
            features: Encoded features of shape (batch_size, seq_len, transformer_width)
            has_cls_token: Whether the sequence includes a CLS token at the last position
        """
        batch_size = biosignals.shape[0]
        
        # Apply convolutional layers
        x = self.conv_layers(biosignals)  # (batch_size, conv_dim, conv_length)
        
        # Transpose to (batch_size, conv_length, conv_dim)
        x = x.transpose(1, 2)
        
        # Project to transformer dimension
        x = self.proj_conv_to_transformer(x)  # (batch_size, conv_length, transformer_width)
        
        # Add positional embeddings
        x = x + self.pos_embed
        
        # Add class token only if needed for pooling
        # For consistency with causal text encoder, append CLS token (not prepend)
        if self.pool_type in ['cls', 'attn']:
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            x = torch.cat([x, cls_tokens], dim=1)  # (batch_size, conv_length + 1, transformer_width)
            has_cls_token = True
        else:
            has_cls_token = False
        
        # Apply transformer layers
        for layer in self.transformer_layers:
            x = layer(x)
            
        # Apply final layer norm
        x = self.ln_final(x)
        
        return x, has_cls_token


class TransformerBlock(nn.Module):
    """Transformer block with self-attention and MLP"""
    
    def __init__(
        self, 
        width: int, 
        heads: int, 
        mlp_ratio: float = 4.0,
        act_layer=QuickGELU,
        norm_layer=LayerNorm,
        dropout: float = 0.1
    ):
        super().__init__()
        self.attention = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
        self.ln_1 = norm_layer(width)
        self.mlp = nn.Sequential(
            nn.Linear(width, int(width * mlp_ratio)),
            act_layer(),
            nn.Dropout(dropout),
            nn.Linear(int(width * mlp_ratio), width),
            nn.Dropout(dropout)
        )
        self.ln_2 = norm_layer(width)
        
    def forward(self, x):
        # Self-attention
        attn_out, _ = self.attention(x, x, x)
        x = x + attn_out
        x = self.ln_1(x)
        
        # MLP
        mlp_out = self.mlp(x)
        x = x + mlp_out
        x = self.ln_2(x)
        
        return x


class AttnPooler(nn.Module):
    """
    CoCa-style attentional pooler.
    A small multi-head attention layer with n_query learned queries (Q),
    and the encoder sequence as both K and V. This lets us:
      - n_query = 1  => global embedding for contrastive loss
      - n_query = N  => compressed token set for decoder cross-attention
    Ref: CoCa uses task-specific attentional pooling with nquery=1 for contrastive
    and nquery=256 for generative objectives.  [oai_citation:2‡Medium](https://medium.com/%40arithmancylabs/coca-contrastive-captioners-are-image-textfoundation-models-324022377630?utm_source=chatgpt.com)
    """
    def __init__(self, dim: int, num_heads: int, n_query: int):
        super().__init__()
        self.n_query = n_query
        self.query_tokens = nn.Parameter(torch.randn(1, n_query, dim) * 0.02)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            batch_first=True
        )

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        """
        x_seq: (B, L, D)
        returns:
            pooled: (B, n_query, D)
        """
        B = x_seq.size(0)
        q = self.query_tokens.expand(B, -1, -1)  # (B, n_query, D)
        pooled, _ = self.attn(q, x_seq, x_seq)   # pooled attends over all tokens
        return pooled  # (B, n_query, D)


class PureTransformerBiosignalsEncoder(BaseBiosignalsEncoder):
    """
    Pure Transformer encoder for biosignals with channel+temporal attention.

    Updated to use CoCa-style task-specific attentional pooling:
    - contrastive_pooler (n_query=1) → 1 global token for contrastive / CLS
    - decoder_pooler (n_query=N_dec) → small set of summary tokens for text decoder

    We still:
      1. Patch each channel independently
      2. Alternate channel-attn and temporal-attn in DualTransformerBlocks (factorized attention)
      3. Keep (B, C, T, D) internally (cheap attention along channel or time separately)
      4. Flatten to (B, C*T, D) only at the end
      5. Run two poolers:
          - 1-query pooler -> global token
          - multi-query pooler -> decoder tokens
      6. Append the 1-query pooled token to the end of x_seq so BaseBiosignalsEncoder
         can keep using pool_type='cls' or 'attn' the same way.
      7. Save the multi-query pooled tokens so, when output_tokens=True, we can hand
         them to the text decoder instead of the full ~C*T sequence.

    This mirrors CoCa's "task-specific attentional pooling," where the same encoder
    supports both contrastive global alignment and caption-style generation with
    minimal extra cost.  [oai_citation:3‡Medium](https://medium.com/%40arithmancylabs/coca-contrastive-captioners-are-image-textfoundation-models-324022377630?utm_source=chatgpt.com)
    """

    def __init__(
        self,
        biosignals_cfg: BiosignalsCfg,
        embed_dim: int = 512,
        output_tokens: bool = False,
        cast_dtype: Optional[torch.dtype] = None
    ):
        super().__init__(
            biosignals_cfg=biosignals_cfg,
            embed_dim=embed_dim,
            output_tokens=output_tokens,
            transformer_width=biosignals_cfg.transformer_width,
            cast_dtype=cast_dtype
        )

        # --- Sanity checks for RoPE dimensions ---
        assert biosignals_cfg.transformer_width % biosignals_cfg.transformer_heads == 0, (
            f"transformer_width ({biosignals_cfg.transformer_width}) must be divisible by "
            f"transformer_heads ({biosignals_cfg.transformer_heads})"
        )
        head_dim = biosignals_cfg.transformer_width // biosignals_cfg.transformer_heads
        assert head_dim % 2 == 0, (
            f"head_dim ({head_dim}) must be even for RoPE. "
            f"Got transformer_width={biosignals_cfg.transformer_width}, "
            f"transformer_heads={biosignals_cfg.transformer_heads}"
        )

        # 1. Channel patching (Conv1d tokenizer per channel)
        self.patching = ChannelPatching(
            patch_size=biosignals_cfg.patch_size,
            conv_embed_dim=biosignals_cfg.conv_embed_dim,
            num_channels=biosignals_cfg.input_channels
        )

        # number of temporal patches per channel
        self.num_patches = biosignals_cfg.signal_length // biosignals_cfg.patch_size

        # 2. Project patch embeddings to transformer_width
        self.embed_projection = nn.Linear(
            biosignals_cfg.conv_embed_dim,
            biosignals_cfg.transformer_width
        )

        # 2a. Channel ID embedding (categorical channel identity)
        self.channel_id_embed = nn.Embedding(
            num_embeddings=biosignals_cfg.input_channels,
            embedding_dim=biosignals_cfg.transformer_width,
        )

        # 3. Shared learnable RoPE for channel attention (optional)
        if biosignals_cfg.share_channel_rope:
            shared_head_dim = biosignals_cfg.transformer_width // biosignals_cfg.transformer_heads
            self.shared_channel_rope = RotaryEmbedding(
                dim=shared_head_dim,
                theta=10000,
                learned_freq=True  # learnable for channel axis
            )
        else:
            self.shared_channel_rope = None

        # 4. Dual-axis Transformer blocks (channel attention + temporal attention)
        self.transformer_blocks = nn.ModuleList([
            DualTransformerBlock(
                embed_dim=biosignals_cfg.transformer_width,
                num_heads=biosignals_cfg.transformer_heads,
                num_temporal_layers=biosignals_cfg.num_temporal_layers,
                dropout=biosignals_cfg.dropout,
                mlp_ratio=biosignals_cfg.mlp_ratio,
                num_channels=biosignals_cfg.input_channels,
                activation=biosignals_cfg.activation,
                norm_type=biosignals_cfg.norm_type,
                mlp_bias=biosignals_cfg.mlp_bias,
                shared_channel_rope=self.shared_channel_rope if biosignals_cfg.share_channel_rope else None
            ) for _ in range(biosignals_cfg.transformer_layers)
        ])

        # 5. Final norm
        norm_layer = (
            LayerNormFp32 if cast_dtype in (torch.float16, torch.bfloat16) else LayerNorm
        )
        if biosignals_cfg.norm_type == "rmsnorm":
            self.ln_final = RMSNorm(biosignals_cfg.transformer_width)
        else:
            self.ln_final = norm_layer(biosignals_cfg.transformer_width)

        # 6. CoCa-style attentional poolers
        #    - contrastive_pooler: n_query = 1 for global CLS token (contrastive head)
        #    - decoder_pooler: n_query = decoder_tokens (e.g. 32) for compressed memory
        #
        # We'll add a new config field on BiosignalsCfg: decoder_tokens (int, default 32).
        n_decoder_tokens = getattr(biosignals_cfg, "decoder_tokens", 32)

        self.contrastive_pooler = AttnPooler(
            dim=biosignals_cfg.transformer_width,
            num_heads=biosignals_cfg.transformer_heads,
            n_query=1
        )

        self.decoder_pooler = AttnPooler(
            dim=biosignals_cfg.transformer_width,
            num_heads=biosignals_cfg.transformer_heads,
            n_query=n_decoder_tokens
        )


    def _encode(self, biosignals: torch.Tensor):
        """
        Returns:
            features: (B, N_dec + 1, D)
                first N_dec tokens  = pooled decoder tokens
                last token          = global pooled token (contrastive CLS)
            has_cls_token: True
        """
        B = biosignals.shape[0]
        device = biosignals.device

        # 1. Patch per channel -> (B, C, T, conv_dim)
        x = self.patching(biosignals)

        # 2. Project to model dim -> (B, C, T, D)
        x = self.embed_projection(x)

        # 2a. Add channel ID embedding
        _, C, T, D = x.shape
        channel_ids = torch.arange(C, device=device)              # (C,)
        channel_bias = self.channel_id_embed(channel_ids)         # (C, D)
        channel_bias = channel_bias.view(1, C, 1, D).expand(B, C, T, D)
        x = x + channel_bias

        # 3. Temporal RoPE positions
        pos_ids = torch.arange(self.num_patches, device=device)   # (T,)

        # 4. Dual-axis transformer blocks (channel-attn + temporal-attn)
        for block in self.transformer_blocks:
            x = block(x, temporal_position_ids=pos_ids)            # stays (B, C, T, D)

        # 5. Final norm
        x = self.ln_final(x)                                      # (B, C, T, D)

        # 6. Flatten channels×time to a sequence for pooling (not for decoder!)
        x_seq = x.reshape(B, C * T, D)                            # (B, L, D) with L = C*T

        # 7. Task-specific attentional pooling (CoCa-style)
        # contrastive_pooler: n_query=1  -> global_token (B,1,D)
        # decoder_pooler:    n_query=Nd -> dec_tokens    (B,Nd,D)
        global_token = self.contrastive_pooler(x_seq)             # (B, 1, D)
        dec_tokens   = self.decoder_pooler(x_seq)                 # (B, N_dec, D)

        # 8. Build final feature sequence:
        #    [decoder tokens..., global token] so that:
        #    - features[:, :-1] = dec_tokens (for decoder cross-attn)
        #    - features[:, -1]  = global_token (for contrastive / CLS pooling)
        features = torch.cat([dec_tokens, global_token], dim=1)   # (B, N_dec+1, D)

        has_cls_token = True
        return features, has_cls_token


class SignalReconstructionDecoder(nn.Module):
    """
    Lightweight transformer decoder for signal reconstruction.
    Uses 2-3 transformer encoder layers + final MLP to reconstruct biosignals.
    Note: Uses TransformerEncoder (self-attention only) since we don't need cross-attention.
    """
    
    def __init__(
        self,
        input_dim: int = 768,
        num_layers: int = 2,
        num_heads: int = 4,  # Reduced from 8 for efficiency
        output_channels: int = 10,
        output_length: int = 1920,
    ):
        super().__init__()
        
        # Transformer encoder layers (self-attention + FFN)
        # Using 2x feedforward (instead of 4x) for lighter decoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=num_heads,
            dim_feedforward=input_dim * 2,  # 1536 for input_dim=768
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Final MLP to project to signal space
        # Reduced intermediate dimension for efficiency
        self.to_signal = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, output_channels * output_length),
        )
        
        self.output_channels = output_channels
        self.output_length = output_length
    
    def forward(self, encoder_features):
        """
        Args:
            encoder_features: (B, seq_len, input_dim) - unprojected encoder features
        Returns:
            reconstructed: (B, output_channels, output_length)
        """
        B = encoder_features.shape[0]
        
        # Self-attention on encoder features
        decoded = self.transformer(encoder_features)  # (B, seq_len, dim)
        
        # Global average pooling
        pooled = decoded.mean(dim=1)  # (B, dim)
        
        # Project to signal space
        signal_flat = self.to_signal(pooled)  # (B, output_channels * output_length)
        
        # Reshape to signal format
        signal = signal_flat.reshape(B, self.output_channels, self.output_length)
        
        return signal


class BiosignalsCoCa(nn.Module):
    """
    CoCa model adapted for biosignals-text contrastive learning.
    Replaces the vision tower with a biosignals encoder.
    
    Supports two decoder types:
        - "cross_attention": Separate cross-attention between text and biosignals (default CoCa)
        - "concat": Concatenate biosignals and text tokens with prefix-causal masking
    """
    
    def __init__(
            self,
            embed_dim,
            multimodal_cfg: MultimodalCfg,
            text_cfg: CLIPTextCfg,
            biosignals_cfg: BiosignalsCfg,
            quick_gelu: bool = False,
            init_logit_scale: float = np.log(1 / 0.07),
            init_logit_bias: Optional[float] = None,
            nonscalar_logit_scale: bool = False,
            cast_dtype: Optional[torch.dtype] = None,
            pad_id: int = 0,
            decoder_type: str = "cross_attention",
            num_caption_channels: int = 12,  # Number of channel/modality embeddings (22 for channels, 4 for modalities)
            prefix_len: int = 0,
            use_signal_decoder: bool = False,  # NEW: Enable signal reconstruction
    ):
        super().__init__()
        multimodal_cfg = MultimodalCfg(**multimodal_cfg) if isinstance(multimodal_cfg, dict) else multimodal_cfg
        text_cfg = CLIPTextCfg(**text_cfg) if isinstance(text_cfg, dict) else text_cfg
        biosignals_cfg = BiosignalsCfg(**biosignals_cfg) if isinstance(biosignals_cfg, dict) else biosignals_cfg

        self.decoder_type = decoder_type
        self.num_channels = num_caption_channels
        self.use_signal_decoder = use_signal_decoder
        
        # Debug logging for channel configuration
        import logging
        logging.info(f"BiosignalsCoCa initialized with num_caption_channels={num_caption_channels}, prefix_len={prefix_len}")
        if use_signal_decoder:
            logging.info(f"Signal reconstruction decoder enabled")

        self.text = _build_text_tower(
            embed_dim=embed_dim,
            text_cfg=text_cfg,
            quick_gelu=quick_gelu,
            cast_dtype=cast_dtype,
        )

        vocab_size = (
            self.text.vocab_size  # for hf models
            if hasattr(text_cfg, "hf_model_name") and text_cfg.hf_model_name is not None
            else text_cfg.vocab_size
        )
        
        # Replace visual tower with biosignals tower
        self.biosignals = _build_signal_tower(
            embed_dim=embed_dim,
            signal_cfg=biosignals_cfg,
            output_tokens=True,  # Need tokens for multimodal decoder
            cast_dtype=cast_dtype,
        )

        self.text_decoder = _build_text_decoder_tower_v2(
            vocab_size,
            multimodal_cfg=multimodal_cfg,
            quick_gelu=quick_gelu,
            cast_dtype=cast_dtype,
            decoder_type=decoder_type,
            prefix_len=prefix_len,
        )

        lshape = [1] if nonscalar_logit_scale else []
        self.logit_scale = nn.Parameter(torch.ones(lshape) * init_logit_scale)
        if init_logit_bias is not None:
            self.logit_bias = nn.Parameter(torch.ones(lshape) * init_logit_bias)
        else:
            self.logit_bias = None
        self.pad_id = pad_id

        self.context_length = multimodal_cfg.context_length
        
        # Learnable channel/modality embeddings
        # num_caption_channels will be 23 for individual channel mode or 5 for modality mode
        # Dimension should match the decoder width (multimodal_cfg.width for text decoder input)
        self.channel_embeddings = nn.Parameter(
            torch.randn(num_caption_channels, multimodal_cfg.width) * 0.02
        )
        
        # Learnable padding embedding for -1 positions
        # This learns to be "neutral" or ignored during training (similar to [PAD] tokens)
        self.padding_embedding = nn.Parameter(
            torch.randn(multimodal_cfg.width) * 0.02
        )
        
        self.decoder_width = multimodal_cfg.width
        
        # Optional signal reconstruction decoder
        if use_signal_decoder:
            self.signal_decoder = SignalReconstructionDecoder(
                input_dim=biosignals_cfg.transformer_width,
                num_layers=2,  # Lightweight: 2 transformer layers
                num_heads=biosignals_cfg.transformer_heads,
                output_channels=biosignals_cfg.input_channels,
                output_length=biosignals_cfg.signal_length,
            )

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable: bool = True):
        self.biosignals.set_grad_checkpointing(enable)
        self.text.set_grad_checkpointing(enable)
        self.text_decoder.set_grad_checkpointing(enable)

    def lock_text_tower(self, unlocked_layers: int = 0, freeze_layer_norm: bool = True):
        """Lock the text encoder, optionally leaving the last N layers unlocked.
        
        Args:
            unlocked_layers: Number of layers to leave unlocked (from the end)
            freeze_layer_norm: Whether to freeze LayerNorm parameters in locked layers
        """
        if hasattr(self.text, 'lock'):
            # For HFTextEncoder (Pythia, etc.)
            self.text.lock(unlocked_layers, freeze_layer_norm)
            
            # IMPORTANT: Unfreeze newly added token embeddings (e.g., <pad>, <coca_cls>)
            # These were randomly initialized and need to be trained
            if hasattr(self.text, 'original_vocab_size'):
                import logging
                embedding_module = self.text.transformer.get_input_embeddings()
                original_size = self.text.original_vocab_size
                current_size = embedding_module.weight.shape[0]
                
                if current_size > original_size:
                    # Enable gradients for the embedding layer
                    embedding_module.weight.requires_grad = True
                    
                    # Store metadata for optimizer configuration (zero weight decay)
                    self.text._new_token_start_idx = original_size
                    
                    # Get actual embedding size (may be padded for Tensor Cores)
                    actual_embedding_size = embedding_module.weight.shape[0]
                    new_vocab_size = self.text.vocab_size  # Actual number of tokens (not padded)
                    
                    # Register parameter-level hook to mask frozen token gradients
                    # IMPORTANT: This is registered BEFORE DDP wrapping to ensure it persists
                    def _zero_grad_frozen_tokens(grad):
                        """Zero out gradients for old (frozen) tokens and padding, keep only new tokens."""
                        if grad is not None:
                            # Zero out pretrained tokens [0:original_size]
                            grad[:original_size] = 0
                            # Zero out padding tokens [new_vocab_size:actual_embedding_size]
                            if actual_embedding_size > new_vocab_size:
                                grad[new_vocab_size:] = 0
                        return grad
                    
                    embedding_module.weight.register_hook(_zero_grad_frozen_tokens)
                    
                    num_new_tokens = new_vocab_size - original_size
                    num_padding_tokens = actual_embedding_size - new_vocab_size
                    logging.info(f"Embedding layer configuration:")
                    logging.info(f"  Trainable new tokens: {num_new_tokens} (indices {original_size}:{new_vocab_size})")
                    logging.info(f"  Frozen pretrained tokens: {original_size} (indices 0:{original_size})")
                    if num_padding_tokens > 0:
                        logging.info(f"  Frozen padding tokens: {num_padding_tokens} (indices {new_vocab_size}:{actual_embedding_size})")
                    logging.info(f"  Total embedding size: {actual_embedding_size}")
                    logging.info(f"Registered gradient masking hook before DDP wrapping")
                    logging.info(f"NOTE: Optimizer uses weight_decay=0 for embedding layer")
        else:
            # For standard TextTransformer
            assert False, "BiosignalsCoCa does not support locking standard TextTransformer"
            from .transformer import lock_text_tower
            lock_text_tower(self, unlocked_layers)

    def _encode_biosignals(self, biosignals, normalize: bool = True):
        biosignals_latent, tokens_embs = self.biosignals(biosignals)
        biosignals_latent = F.normalize(biosignals_latent, dim=-1) if normalize else biosignals_latent
        return biosignals_latent, tokens_embs

    def _encode_text(self, text, normalize: bool = True):
        text_latent, token_emb = self.text(text)
        text_latent = F.normalize(text_latent, dim=-1) if normalize else text_latent
        return text_latent, token_emb

    def encode_image(self, biosignals, normalize: bool = True):
        biosignals_latent, _ = self._encode_biosignals(biosignals, normalize=normalize)
        return biosignals_latent

    def encode_text(self, text, normalize: bool = True):
        text_latent, _ = self._encode_text(text, normalize=normalize)
        return text_latent

    def _get_channel_condition_embs(self, channel_indices: torch.Tensor) -> torch.Tensor:
        """Convert channel/modality indices to embeddings with learnable padding.
        
        Args:
            channel_indices: (batch_size, prefix_len) tensor of indices
                - Individual mode: indices into 23 channel embeddings (22 channels + 1 stage_event)
                - Modality mode: indices into 5 modality embeddings (4 modalities + 1 stage_event)
                - Padded with -1 for variable length (uses learnable padding_embedding for -1)
            
        Returns:
            condition_embs: (batch_size, prefix_len, decoder_width)
                Embeddings for all positions. -1 positions use learnable padding_embedding
                that learns to be neutral/ignored during training.
        """
        batch_size, prefix_len = channel_indices.shape
        
        # Create output tensor
        condition_embs = torch.zeros(batch_size, prefix_len, self.decoder_width, 
                                     dtype=self.channel_embeddings.dtype, 
                                     device=self.channel_embeddings.device)
        
        # Create mask for valid (non-padding) indices
        valid_mask = channel_indices >= 0  # (batch_size, prefix_len)
        padding_mask = channel_indices == -1  # (batch_size, prefix_len)
        
        # Gather channel embeddings for valid indices
        # Clamp to 0 for safe indexing (will be overwritten by padding where needed)
        indices_safe = channel_indices.clamp(min=0)
        
        # Expand embeddings for batching
        expanded_embeddings = self.channel_embeddings.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Gather embeddings
        indices_expanded = indices_safe.unsqueeze(-1).expand(-1, -1, self.decoder_width)
        gathered_embs = torch.gather(expanded_embeddings, 1, indices_expanded)
        
        # Fill in valid positions with gathered embeddings
        condition_embs[valid_mask] = gathered_embs[valid_mask]
        
        # Fill in padding positions with learnable padding embedding
        if padding_mask.any():
            # Broadcast padding_embedding to all padding positions
            condition_embs[padding_mask] = self.padding_embedding
        
        return condition_embs
    
    def forward(
            self,
            biosignals,
            text: Optional[torch.Tensor] = None,
            biosignals_latent: Optional[torch.Tensor] = None,
            biosignals_embs: Optional[torch.Tensor] = None,

            channel_indices: Optional[torch.Tensor] = None,
            output_labels: bool = True,
    ):
        """Forward pass for BiosignalsCoCa model.
        
        Args:
            biosignals: Input biosignals tensor
            text: Optional text token ids
            biosignals_latent: Optional pre-computed biosignals latent features
            biosignals_embs: Optional pre-computed biosignals token embeddings

            channel_indices: Optional (batch_size, num_selected_channels) tensor of channel indices
                Used to select channel-specific condition embeddings. If provided, overrides condition_embs.
            output_labels: Whether to output labels for loss computation
        """
        if biosignals_latent is None or biosignals_embs is None:
            biosignals_latent, biosignals_embs = self._encode_biosignals(biosignals)

        if text is None:
            return {"image_features": biosignals_latent, "image_embs": biosignals_embs}

        text_latent, token_embs = self._encode_text(text)

        # FIXME this isn't an ideal solution, would like to improve -RW
        labels: Optional[torch.Tensor] = text[:, 1:] if output_labels else None
        if output_labels:
            # align text_embs and thus logits with labels for teacher-forcing caption loss
            token_embs = token_embs[:, :-1]
        
        # Convert channel indices to condition embeddings if provided
        if channel_indices is not None:
            condition_embs = self._get_channel_condition_embs(channel_indices)
        else:
            condition_embs = None

        logits = self.text_decoder(biosignals_embs, token_embs, condition_embs=condition_embs)
        out_dict = {
            "image_features": biosignals_latent,
            "text_features": text_latent,
            "logits": logits,
            "logit_scale": self.logit_scale.exp()
        }
        if labels is not None:
            out_dict["labels"] = labels
        if self.logit_bias is not None:
            out_dict["logit_bias"] = self.logit_bias
        
        # Optional signal reconstruction
        if self.use_signal_decoder:
            reconstructed_signal = self.signal_decoder(biosignals_embs)
            out_dict["reconstructed_signal"] = reconstructed_signal
            out_dict["original_signal"] = biosignals
        
        return out_dict

    def generate(
        self,
        biosignals,
        text=None,
        seq_len=30,
        max_seq_len=256,
        temperature=1.,
        generation_type="beam_search",
        top_p=0.1,
        top_k=1,
        pad_token_id=None,
        eos_token_id=None,
        sot_token_id=None,
        num_beams=6,
        num_beam_groups=3,
        min_seq_len=5,
        stopping_criteria=None,
        repetition_penalty=1.0,
        fixed_output_length=False,
        condition_embs=None,
        channel_indices=None,
    ):
# taking many ideas and components from HuggingFace GenerationMixin
        # https://huggingface.co/docs/transformers/main/en/main_classes/text_generation
        assert _has_transformers, "Please install transformers for generate functionality. `pip install transformers`."
        assert seq_len > min_seq_len, "seq_len must be larger than min_seq_len"
        device = biosignals.device
        
        # Note: condition_embs parameter is for backward compatibility
        # We pass channel_indices directly to forward(), which handles the conversion internally

        with torch.no_grad():
            sot_token_id = _token_to_tensor(sot_token_id, device=device)
            eos_token_id = _token_to_tensor(eos_token_id, device=device)
            pad_token_id = pad_token_id
            logit_processor = LogitsProcessorList(
                [
                    MinLengthLogitsProcessor(min_seq_len, eos_token_id),
                    RepetitionPenaltyLogitsProcessor(repetition_penalty),
                ]
            )

            if stopping_criteria is None:
                stopping_criteria = [MaxLengthCriteria(max_length=seq_len)]
            stopping_criteria = StoppingCriteriaList(stopping_criteria)

            if generation_type == "beam_search":
                output = self._generate_beamsearch(
                    biosignals_inputs=biosignals,
                    pad_token_id=pad_token_id,
                    eos_token_id=eos_token_id,
                    sot_token_id=sot_token_id,
                    num_beams=num_beams,
                    num_beam_groups=num_beam_groups,
                    min_seq_len=min_seq_len,
                    stopping_criteria=stopping_criteria,
                    logit_processor=logit_processor,
                    channel_indices=channel_indices,
                )
                if fixed_output_length and output.shape[1] < seq_len:
                    pad_len = seq_len - output.shape[1]
                    return torch.cat((
                            output,
                            torch.ones(output.shape[0], pad_len, device=device, dtype=output.dtype) * pad_token_id
                        ),
                        dim=1
                    )
                return output

            elif generation_type == "top_p":
                logit_warper = GENERATION_TYPES[generation_type](top_p)
            elif generation_type == "top_k":
                logit_warper = GENERATION_TYPES[generation_type](top_k)
            else:
                raise ValueError(
                    f"generation_type has to be one of "
                    f"{'| ' + ' | '.join(list(GENERATION_TYPES.keys())) + ' |'}."
                )

            biosignals_latent, biosignals_embs = self._encode_biosignals(biosignals)

            if text is None:
                text = torch.ones((biosignals.shape[0], 1), device=device, dtype=torch.long) * sot_token_id

            was_training = self.training
            num_dims = len(text.shape)

            if num_dims == 1:
                text = text[None, :]

            self.eval()
            out = text

            while True:
                x = out[:, -max_seq_len:]
                cur_len = x.shape[1]
                logits = self(
                    biosignals,
                    x,
                    biosignals_latent=biosignals_latent,
                    biosignals_embs=biosignals_embs,
                    channel_indices=channel_indices,
                    output_labels=False,
                )["logits"][:, -1]
                mask = (out[:, -1] == eos_token_id) | (out[:, -1] == pad_token_id)
                sample = torch.ones((out.shape[0], 1), device=device, dtype=torch.long) * pad_token_id

                if mask.all():
                    if not fixed_output_length:
                        break
                else:
                    logits = logits[~mask, :]
                    filtered_logits = logit_processor(x[~mask, :], logits)
                    filtered_logits = logit_warper(x[~mask, :], filtered_logits)
                    probs = F.softmax(filtered_logits / temperature, dim=-1)

                    if (cur_len + 1 == seq_len):
                        sample[~mask, :] = torch.ones((sum(~mask), 1), device=device, dtype=torch.long) * eos_token_id
                    else:
                        sample[~mask, :] = torch.multinomial(probs, 1)

                out = torch.cat((out, sample), dim=-1)

                cur_len += 1

                if all(stopping_criteria(out, None)):
                    break

            if num_dims == 1:
                out = out.squeeze(0)

            self.train(was_training)
            return out

    def _generate_beamsearch(
            self,
            biosignals_inputs,
            pad_token_id=None,
            eos_token_id=None,
            sot_token_id=None,
            num_beams=6,
            num_beam_groups=3,
            min_seq_len=5,
            stopping_criteria=None,
            logit_processor=None,
            logit_warper=None,
            channel_indices=None,
    ):
        device = biosignals_inputs.device
        batch_size = biosignals_inputs.shape[0]
        biosignals_inputs = torch.repeat_interleave(biosignals_inputs, num_beams, dim=0)
        biosignals_latent, biosignals_embs = self._encode_biosignals(biosignals_inputs)
        
        # Repeat channel indices for beam search if provided
        # forward() will convert them to condition embeddings internally
        if channel_indices is not None:
            channel_indices = torch.repeat_interleave(channel_indices, num_beams, dim=0)

        input_ids = torch.ones((batch_size * num_beams, 1), device=device, dtype=torch.long)
        input_ids = input_ids * sot_token_id
        beam_scorer = BeamSearchScorer(
            batch_size=batch_size,
            num_beams=num_beams,
            device=device,
            num_beam_groups=num_beam_groups,
        )
        # instantiate logits processors
        logits_processor = (
            LogitsProcessorList([MinLengthLogitsProcessor(min_seq_len, eos_token_id=eos_token_id)])
            if logit_processor is None
            else logit_processor
        )

        num_beams = beam_scorer.num_beams
        num_beam_groups = beam_scorer.num_beam_groups
        num_sub_beams = num_beams // num_beam_groups
        batch_size = len(beam_scorer._beam_hyps) // num_beam_groups
        batch_beam_size, cur_len = input_ids.shape
        beam_indices = None

        if num_beams * batch_size != batch_beam_size:
            raise ValueError(
                f"Batch dimension of `input_ids` should be {num_beams * batch_size}, but is {batch_beam_size}."
            )

        beam_scores = torch.full((batch_size, num_beams), -1e9, dtype=torch.float, device=device)
        # initialise score of first beam of each group with 0 and the rest with 1e-9. This ensures that the beams in
        # the same group don't produce same tokens everytime.
        beam_scores[:, ::num_sub_beams] = 0
        beam_scores = beam_scores.view((batch_size * num_beams,))

        while True:

            # predicted tokens in cur_len step
            current_tokens = torch.zeros(batch_size * num_beams, dtype=input_ids.dtype, device=device)

            # indices which will form the beams in the next time step
            reordering_indices = torch.zeros(batch_size * num_beams, dtype=torch.long, device=device)

            # do one decoder step on all beams of all sentences in batch
            model_inputs = prepare_inputs_for_generation(input_ids=input_ids, biosignals_inputs=biosignals_inputs)
            outputs = self(
                model_inputs['biosignals'],
                model_inputs['text'],
                biosignals_latent=biosignals_latent,
                biosignals_embs=biosignals_embs,
                channel_indices=channel_indices,
                output_labels=False,
            )

            for beam_group_idx in range(num_beam_groups):
                group_start_idx = beam_group_idx * num_sub_beams
                group_end_idx = min(group_start_idx + num_sub_beams, num_beams)
                group_size = group_end_idx - group_start_idx

                # indices of beams of current group among all sentences in batch
                batch_group_indices = []

                for batch_idx in range(batch_size):
                    batch_group_indices.extend(
                        [batch_idx * num_beams + idx for idx in range(group_start_idx, group_end_idx)]
                    )
                group_input_ids = input_ids[batch_group_indices]

                # select outputs of beams of currentg group only
                next_token_logits = outputs['logits'][batch_group_indices, -1, :]
                vocab_size = next_token_logits.shape[-1]

                next_token_scores_processed = logits_processor(
                    group_input_ids, next_token_logits, current_tokens=current_tokens, beam_group_idx=beam_group_idx
                )
                next_token_scores = next_token_scores_processed + beam_scores[batch_group_indices].unsqueeze(-1)
                next_token_scores = next_token_scores.expand_as(next_token_scores_processed)

                # reshape for beam search
                next_token_scores = next_token_scores.view(batch_size, group_size * vocab_size)

                next_token_scores, next_tokens = torch.topk(
                    next_token_scores, 2 * group_size, dim=1, largest=True, sorted=True
                )

                next_indices = torch.div(next_tokens, vocab_size, rounding_mode="floor")
                next_tokens = next_tokens % vocab_size

                # stateless
                process_beam_indices = sum(beam_indices, ()) if beam_indices is not None else None
                beam_outputs = beam_scorer.process(
                    group_input_ids,
                    next_token_scores,
                    next_tokens,
                    next_indices,
                    pad_token_id=pad_token_id,
                    eos_token_id=eos_token_id,
                    beam_indices=process_beam_indices,
                    group_index=beam_group_idx,
                )
                beam_scores[batch_group_indices] = beam_outputs["next_beam_scores"]
                beam_next_tokens = beam_outputs["next_beam_tokens"]
                beam_idx = beam_outputs["next_beam_indices"]

                input_ids[batch_group_indices] = group_input_ids[beam_idx]
                group_input_ids = torch.cat([group_input_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1)
                current_tokens[batch_group_indices] = group_input_ids[:, -1]

                # (beam_idx // group_size) -> batch_idx
                # (beam_idx % group_size) -> offset of idx inside the group
                reordering_indices[batch_group_indices] = (
                    num_beams * torch.div(beam_idx, group_size, rounding_mode="floor") + group_start_idx + (beam_idx % group_size)
                )

            input_ids = torch.cat([input_ids, current_tokens.unsqueeze(-1)], dim=-1)

            # increase cur_len
            cur_len = cur_len + 1
            if beam_scorer.is_done or all(stopping_criteria(input_ids, None)):
                break

        final_beam_indices = sum(beam_indices, ()) if beam_indices is not None else None
        sequence_outputs = beam_scorer.finalize(
            input_ids,
            beam_scores,
            next_tokens,
            next_indices,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            max_length=stopping_criteria.max_length,
            beam_indices=final_beam_indices,
        )
        return sequence_outputs['sequences']


def prepare_inputs_for_generation(input_ids, biosignals_inputs, past=None, **kwargs):
    if past:
        input_ids = input_ids[:, -1].unsqueeze(-1)

    attention_mask = kwargs.get("attention_mask", None)
    position_ids = kwargs.get("position_ids", None)

    if attention_mask is not None and position_ids is None:
        # create position_ids on the fly for batch generation
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
    else:
        position_ids = None
    return {
        "text": input_ids,
        "biosignals": biosignals_inputs,
        "past_key_values": past,
        "position_ids": position_ids,
        "attention_mask": attention_mask,
    }
