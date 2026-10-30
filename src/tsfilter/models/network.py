from __future__ import annotations

import torch
from torch import nn


class FilterPredictor(nn.Module):
    def __init__(
            self,
            n_features: int,
            n_classes: int,
            use_signal: bool = True,
            use_features: bool = True,
            features_hidden_size: int = 32,
            channels: tuple[int, ...] = (16, 32, 64),
            kernel_size: int = 7,
            pool_size: int = 4,
            combined_hidden_size: int = 64,
            batch_norm: bool = True,
            activation_function =  nn.ReLU()
    ):
        super().__init__()
        self.use_signal = use_signal
        self.use_features = use_features

        layers, in_channels = [], 1

        for out_channels in channels:
            layers += [
                # kolko kanálov je na vstupe, kolko ich má byť na výstupe a aby sa neskracoval signál, na každý kraj pridáme polovicu kernelu
                # L_out = L + 2·padding − kernel_size + 1
                nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2),
                nn.BatchNorm1d(out_channels) if batch_norm else nn.Identity(), activation_function,
                nn.MaxPool1d(pool_size)
            ]
            in_channels = out_channels
        layers += [nn.AdaptiveAvgPool1d(1), nn.Flatten()]
        self.signal_branch = nn.Sequential(*layers) if use_signal else None

        # n príznakov, ktoré spájame so skrytou vrstvou x neurónmi, hľadáme vzťahy medzi štastistikami (ak je priemer nízky, ale odchýlka veľka atd.)
        if not use_features:
            self.feature_branch, feature_size = None, 0
        elif features_hidden_size > 0:
            self.feature_branch, feature_size = nn.Sequential(nn.Linear(n_features, features_hidden_size), activation_function), features_hidden_size
        else:
            self.feature_branch, feature_size = nn.Identity(), n_features

        signal_size = channels[-1] if use_signal else 0
        self.classifier = nn.Sequential(
            nn.Dropout(0.3), # nahodne vypneme 30% neuronov pri danom behu, aby sa neučili od konkrétnych susedov
            nn.Linear(signal_size + feature_size, combined_hidden_size), activation_function,
            nn.Linear(combined_hidden_size, n_classes), # vysledná vrstva
        )

    def forward(self, signal: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        parts = []

        if self.use_signal:
            parts.append(self.signal_branch(signal.unsqueeze(1)))
        if self.use_features:
            parts.append(self.feature_branch(features))

        return self.classifier(torch.cat(parts, dim=1))
