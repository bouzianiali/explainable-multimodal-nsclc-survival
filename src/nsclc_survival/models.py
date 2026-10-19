from __future__ import annotations
import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, cin:int, cout:int):
        super().__init__()
        self.net=nn.Sequential(
            nn.Conv3d(cin,cout,3,stride=2,padding=1,bias=False), nn.BatchNorm3d(cout), nn.ReLU(inplace=True),
            nn.Conv3d(cout,cout,3,stride=1,padding=1,bias=False), nn.BatchNorm3d(cout), nn.ReLU(inplace=True))
    def forward(self,x): return self.net(x)

class CTOnlyTeacher(nn.Module):
    def __init__(self,dropout:float=.30):
        super().__init__()
        self.encoder=nn.Sequential(ConvBlock(1,8),ConvBlock(8,16),ConvBlock(16,32),ConvBlock(32,64),ConvBlock(64,96),nn.AdaptiveAvgPool3d(1))
        self.head=nn.Sequential(nn.Flatten(),nn.Dropout(dropout),nn.Linear(96,32),nn.ReLU(inplace=True),nn.Dropout(dropout),nn.Linear(32,1))
    def forward(self,x): return self.head(self.encoder(x)).squeeze(1)

class GatedResidualSurvival(nn.Module):
    """A/B/C share this architecture; ModDrop and anchor are training components."""
    def __init__(self,clinical_dim:int=21,dropout:float=.30):
        super().__init__()
        self.ct_encoder=nn.Sequential(ConvBlock(1,8),ConvBlock(8,16),ConvBlock(16,32),ConvBlock(32,64),ConvBlock(64,96),nn.AdaptiveAvgPool3d(1))
        self.ct_risk_head=nn.Sequential(nn.Flatten(),nn.Dropout(dropout),nn.Linear(96,32),nn.ReLU(inplace=True),nn.Dropout(dropout),nn.Linear(32,1))
        self.ct_feature_proj=nn.Sequential(nn.Linear(96,64),nn.ReLU(inplace=True),nn.LayerNorm(64))
        self.clinical_encoder=nn.Sequential(nn.Linear(clinical_dim,64),nn.ReLU(inplace=True),nn.Dropout(dropout),nn.Linear(64,64),nn.ReLU(inplace=True),nn.LayerNorm(64))
        self.gate=nn.Sequential(nn.Linear(129,64),nn.ReLU(inplace=True),nn.Linear(64,1),nn.Sigmoid())
        self.residual_head=nn.Sequential(nn.Linear(128,64),nn.ReLU(inplace=True),nn.Dropout(dropout),nn.Linear(64,1))
    def forward(self,ct,clinical,availability):
        fmap=self.ct_encoder(ct); flat=torch.flatten(fmap,1)
        rct=self.ct_risk_head(fmap).squeeze(1); zct=self.ct_feature_proj(flat)
        availability=availability.reshape(-1,1).to(dtype=clinical.dtype)
        zcl=self.clinical_encoder(clinical*availability)
        g=self.gate(torch.cat([zct,zcl,availability],1)).squeeze(1)
        rcl=self.residual_head(torch.cat([zct,zcl],1)).squeeze(1)
        final=rct+availability.squeeze(1)*g*rcl
        return final,rct,g,rcl

def parameter_counts():
    ct=CTOnlyTeacher(); gated=GatedResidualSurvival()
    return {'ct_only':sum(p.numel() for p in ct.parameters()),'gated_residual':sum(p.numel() for p in gated.parameters())}
