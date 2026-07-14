from agents.gciql import GCIQLAgent
from agents.gcivl import GCIVLAgent
from agents.ota_flow import OTAAgent_flow
from agents.ota import OTAAgent


agents = dict(
    gciql=GCIQLAgent,
    gcivl=GCIVLAgent,
    ota_flow=OTAAgent_flow,
    ota=OTAAgent,
)
