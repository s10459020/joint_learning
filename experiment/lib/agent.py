from experiment.agents.aspl import ASPLAgent, ASPLCAgent, ASPLGPAgent
from experiment.agents.bc import BCAgent
from experiment.agents.cql import CQLAgent
from experiment.agents.iql import IQLAgent
from experiment.agents.scas import SCASAgent, SCASGPAgent, SCASGPNAgent, SCASNAgent
from experiment.agents.scaspl import (
    SCASPLAgent,
    SCASPLCAgent,
    SCASPLGPAgent,
    SCASPLNAgent,
    SCASPLNCAgent,
)
from experiment.agents.sccc import (
    SCCCAgent,
    SCCCGPAgent,
    SCCCGPNAgent,
    SCCCNAgent,
)
from experiment.agents.td3bc import TD3BCAgent, TD3BCGPAgent, TD3BCPGPAgent, TD3BCPAgent, TD3BCXNAgent, TD3BCXNGPAgent


AGENT_CLASSES = {
    "bc": BCAgent,
    "td3bc": TD3BCAgent,
    "td3bc_xn": TD3BCXNAgent,
    "td3bc_p": TD3BCPAgent,
    "td3bc_xn_gp": TD3BCXNGPAgent,
    "td3bc_p_gp": TD3BCPGPAgent,
    "td3bc_gp": TD3BCGPAgent,
    "iql": IQLAgent,
    "cql": CQLAgent,
    "aspl": ASPLAgent,
    "aspl_c": ASPLCAgent,
    "aspl_gp": ASPLGPAgent,
}

DYNAMIC_AGENT_CLASSES = {
    "scas": SCASAgent,
    "scas_n": SCASNAgent,
    "scas_gp": SCASGPAgent,
    "scas_gpn": SCASGPNAgent,
    "scaspl": SCASPLAgent,
    "scaspl_n": SCASPLNAgent,
    "scaspl_gp": SCASPLGPAgent,
    "scaspl_c": SCASPLCAgent,
    "scaspl_nc": SCASPLNCAgent,
    "sccc": SCCCAgent,
    "sccc_n": SCCCNAgent,
    "sccc_gp": SCCCGPAgent,
    "sccc_gpn": SCCCGPNAgent,
}


def make_agent(agent_id: str, dataset, dynamic=None):
    if agent_id in DYNAMIC_AGENT_CLASSES:
        agent_class = DYNAMIC_AGENT_CLASSES[agent_id]
        agent = agent_class(
            dataset.obs_size,
            dataset.act_size,
            dynamic=dynamic,
            device=dataset.device,
        )
        agent.id = agent_id
        return agent

    agent_class = AGENT_CLASSES[agent_id]
    agent = agent_class(dataset.obs_size, dataset.act_size, device=dataset.device)
    agent.id = agent_id
    return agent
