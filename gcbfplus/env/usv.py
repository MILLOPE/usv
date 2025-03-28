import functools as ft
import pathlib
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from typing import NamedTuple, Tuple, Optional

from ..utils.graph import EdgeBlock, GetGraph, GraphsTuple
from ..utils.typing import Action, AgentState, Array, Cost, Done, Info, Pos2d, Reward, State
from ..utils.utils import merge01
from .base import MultiAgentEnv, RolloutResult
from .obstacle import Obstacle, Rectangle
from .plot import render_video
from .utils import get_lidar, inside_obstacles, get_node_goal_rng

import ipdb

class USV(MultiAgentEnv):
    AGENT = 0
    GOAL = 1
    OBS = 2

    class EnvState(NamedTuple):
        agent: AgentState
        goal: State
        obstacle: Obstacle

        @property
        def n_agent(self) -> int:
            return self.agent.shape[0]

    EnvGraphsTuple = GraphsTuple[State, EnvState]

    PARAMS = {
        "car_radius": 0.05,
        "comm_radius": 0.5,
        "n_rays": 16,
        "obs_len_range": [0.1, 0.6],
        "n_obs": 8,
    }

    def __init__(
            self,
            num_agents: int,
            area_size: float,
            max_step: int = 256,
            max_travel: float = None,
            dt: float = 0.03,
            params: dict = None
    ):
        super(USV, self).__init__(num_agents, area_size, max_step, max_travel, dt, params)
        self.create_obstacles = jax.vmap(Rectangle.create)
        self.enable_stop = True

    @property
    def state_dim(self) -> int:
        return 6  # 从4改为6: [x, y, φ, u, v, r]

    @property
    def node_dim(self) -> int:
        return 3  # indicator: agent: 001, goal: 010, obstacle: 100

    @property
    def edge_dim(self) -> int:
        return 5  # 从4改为5: [dx_body, dy_body, du, dv, dr]

    @property
    def action_dim(self) -> int:
        return 2  # omega, acc

    def reset(self, key: Array) -> GraphsTuple:
        self._t = 0

        # randomly generate obstacles
        obstacle_key, key = jr.split(key, 2)
        obs_pos = jr.uniform(obstacle_key, (self._params["n_obs"], 2), minval=0, maxval=self.area_size)
        length_key, key = jr.split(key, 2)
        obs_len = jr.uniform(
            length_key,
            (self._params["n_obs"], 2),
            minval=self._params["obs_len_range"][0],
            maxval=self._params["obs_len_range"][1],
        )
        theta_key, key = jr.split(key, 2)
        obs_theta = jr.uniform(theta_key, (self._params["n_obs"],), minval=0, maxval=2 * np.pi)
        obstacles = self.create_obstacles(obs_pos, obs_len[:, 0], obs_len[:, 1], obs_theta)

        # randomly generate agent and goal
        pos_key, key = jr.split(key)
        states_pos, goals_pos = get_node_goal_rng(
            pos_key, self.area_size, 2, obstacles, self.num_agents,
            4 * self.params["car_radius"], self.max_travel
        )
        
         # 初始化完整状态向量 [x, y, φ, u, v, r]
        states = jnp.zeros((self.num_agents, 6))
        goals = jnp.zeros((self.num_agents, 6))  
        
        # 添加位置信息 
        states = states.at[:, :2].set(states_pos)
        goals = goals.at[:, :2].set(goals_pos)

        # 随机初始航向
        phi_key, key = jr.split(key)
        initial_phi = jr.uniform(phi_key, (self.num_agents,), 
                                minval=-jnp.pi, maxval=jnp.pi)
        states = states.at[:, 2].set(initial_phi)

        # 设置目标航向（朝向目标点）
        target_phi = jnp.arctan2(goals_pos[:,1]-states_pos[:,1], 
                            goals_pos[:,0]-states_pos[:,0])
        goals = goals.at[:, 2].set(target_phi)

        # 初始化速度（可添加随机小扰动）
        vel_key, key = jr.split(key)
        initial_u = jr.normal(vel_key, (self.num_agents,)) * 0.1
        states = states.at[:, 3].set(initial_u)  # u

        env_states = self.EnvState(states, goals, obstacles)

        assert env_states.agent.shape == (self.num_agents, 6)
        assert env_states.goal.shape == (self.num_agents, 6)
        return self.get_graph(env_states)

    def agent_step_euler(self, agent_states: AgentState, action: Action, stop_mask: Array) -> AgentState:
        assert action.shape == (self.num_agents, self.action_dim)
        assert agent_states.shape == (self.num_agents, self.state_dim)
        x_dot = self.agent_xdot(agent_states, action) * (1 - stop_mask)[:, None]
        n_state_agent_new = agent_states + x_dot * self.dt
        assert n_state_agent_new.shape == (self.num_agents, self.state_dim)
        return self.clip_state(n_state_agent_new)

    def agent_xdot(self, agent_states: AgentState, action: Action) -> AgentState:
        assert action.shape == (self.num_agents, self.action_dim)
        assert agent_states.shape == (self.num_agents, self.state_dim)
        x_dot = jnp.concatenate([
            (jnp.cos(agent_states[:, 2]) * agent_states[:, 3])[:, None],
            (jnp.sin(agent_states[:, 2]) * agent_states[:, 3])[:, None],
            (action[:, 0] * 20.)[:, None],
            (action[:, 1])[:, None]
        ], axis=1)
        assert x_dot.shape == (self.num_agents, self.state_dim)
        return x_dot

    def step(
            self, graph: EnvGraphsTuple, action: Action, get_eval_info: bool = False
    ) -> Tuple[EnvGraphsTuple, Reward, Cost, Done, Info]:
        self._t += 1

        # calculate next graph
        agent_states = graph.type_states(type_idx=0, n_type=self.num_agents)
        goal_states = graph.type_states(type_idx=1, n_type=self.num_agents)
        obstacles = graph.env_states.obstacle
        action = self.clip_action(action)

        assert action.shape == (self.num_agents, self.action_dim)
        assert agent_states.shape == (self.num_agents, self.state_dim)

        stop_mask = self.stop_mask(graph)
        if not self.enable_stop:
            # If stopping is not enabled, then set stop_mask to always be 0.
            stop_mask = 0 * stop_mask
        next_agent_states = self.agent_step_euler(agent_states, action, stop_mask)

        # the episode ends when reaching max_episode_steps
        done = jnp.array(False)

        # compute reward and cost
        reward = jnp.zeros(()).astype(jnp.float32)
        reward -= (jnp.linalg.norm(action - self.u_ref(graph), axis=1) ** 2).mean()
        cost = self.get_cost(graph)

        assert reward.shape == tuple()
        assert cost.shape == tuple()
        assert done.shape == tuple()

        next_state = self.EnvState(next_agent_states, goal_states, obstacles)

        return self.get_graph(next_state), reward, cost, done, {}

    def get_cost(self, graph: EnvGraphsTuple) -> Cost:
        agent_states = graph.type_states(type_idx=0, n_type=self.num_agents)
        obstacles = graph.env_states.obstacle

        # collision between agents
        agent_pos = agent_states[:, :2]
        dist = jnp.linalg.norm(jnp.expand_dims(agent_pos, 1) - jnp.expand_dims(agent_pos, 0), axis=-1)
        dist += jnp.eye(self.num_agents) * 1e6
        collision = (self._params["car_radius"] * 2 > dist).any(axis=1)
        cost = collision.mean()

        # collision between agents and obstacles
        collision = inside_obstacles(agent_pos, obstacles, r=self._params["car_radius"])
        cost += collision.mean()

        return cost

    def render_video(
        self, rollout: RolloutResult, video_path: pathlib.Path, Ta_is_unsafe=None, viz_opts: dict = None, dpi: int = 80, **kwargs
    ) -> None:
        render_video(
            rollout=rollout,
            video_path=video_path,
            side_length=self.area_size,
            dim=2,
            n_agent=self.num_agents,
            n_rays=self.params["n_rays"],
            r=self.params["car_radius"],
            Ta_is_unsafe=Ta_is_unsafe,
            viz_opts=viz_opts,
            dpi=dpi,
            **kwargs
        )

    def edge_blocks(self, state: EnvState, lidar_data: State) -> list[EdgeBlock]:
        n_hits = self._params["n_rays"] * self.num_agents

        agent_states = state.agent  # [N,6]
        assert agent_states.shape[1] == 6 
        goal_states = state.goal    # [N,6]
        agent_pos = state.agent[:, :2]

        # ===== 辅助函数 =====
        def get_body_relative(pos_diff, phi):
            """将全局位置差转换到本体坐标系"""
            R_inv = jnp.array([[jnp.cos(phi), jnp.sin(phi)],
                            [-jnp.sin(phi), jnp.cos(phi)]])
            return jnp.dot(R_inv, pos_diff)

        # ===== Agent-Agent 连接 =====
        # 本体坐标系相对位置
        global_pos_diff = agent_states[:, None, :2] - agent_states[None, :, :2]  # [i,j,2]
        body_pos_diff = global_pos_diff
        # def process_row(g_row, phi_i):
        #     return jax.vmap(lambda pos_diff: get_body_relative(pos_diff, phi_i))(g_row)

        # body_pos_diff = jax.vmap(process_row, in_axes=(0, 0))(
        #     global_pos_diff,  # shape (num_agents, num_agents, 2)
        #     agent_states[:, 2]  # shape (num_agents,)
        # )

        # 本体速度差
        u = agent_states[:, 3]  # [N]
        v = agent_states[:, 4]
        r = agent_states[:, 5]
        du = u[:, None] - u[None, :]  # [i,j]
        dv = v[:, None] - v[None, :]
        dr = r[:, None] - r[None, :]

        # 构建边特征 [dx_body, dy_body, du, dv, dr]
        agent_agent_feats = jnp.concatenate([
            body_pos_diff,          # [i,j,2]
            du[..., None],          # [i,j,1]
            dv[..., None],          # [i,j,1]
            dr[..., None]           # [i,j,1]
        ], axis=-1)  # [i,j,5]

        # 通信掩码
        dist = jnp.linalg.norm(global_pos_diff, axis=-1)
        mask = jnp.less(dist, self._params["comm_radius"])
        mask = mask & (1 - jnp.eye(self.num_agents, dtype=bool))  # 排除自连接
        
        id_agent = jnp.arange(self.num_agents)
        agent_agent = EdgeBlock(agent_agent_feats, mask, id_agent, id_agent)

        # ===== Agent-Goal 连接 =====
        agent_goal_feats = []
        for i in range(self.num_agents):
            # 本体坐标系位置差
            global_diff = agent_states[i, :2] - goal_states[i, :2]
            body_diff = get_body_relative(global_diff, agent_states[i, 2])
            
            # 速度特征
            u_rel = u[i]  # 本体速度直接使用
            v_rel = v[i]
            
            # 构建特征 [dx_body, dy_body, u, v, r]
            feat = jnp.concatenate([
                body_diff, 
                jnp.array([u_rel, v_rel, r[i]])
            ])
            agent_goal_feats.append(feat[None, None, :])  # [1,1,5]

        id_goal = jnp.arange(self.num_agents, 2*self.num_agents)
        agent_goal = [
            EdgeBlock(feat, jnp.ones((1,1)), id_agent[i][None], id_goal[i][None])
            for i, feat in enumerate(agent_goal_feats)
        ]
        assert len(agent_goal) == self.num_agents
        # assert agent_goal[0].feats.shape == (1, 1, 5)

        # ===== Agent-Obstacle 连接 =====
        # Agent-Obstacle连接
        agent_obs_edges = []
        id_obs = jnp.arange(2*self.num_agents, 2*self.num_agents + n_hits)
        for i in range(self.num_agents):
            # 激光数据转换到本体坐标系
            id_hits = jnp.arange(i * self._params["n_rays"], (i + 1) * self._params["n_rays"])
            pos_diff = agent_states[i, :2] - lidar_data[id_hits, :2]
            
            current_agent = state.agent[i]
            u = current_agent[3]  
            v = current_agent[4]  
            r = current_agent[5]  
            speed_feat = jnp.array([u, v, r])[None, :]  # shape (1,3)
            speed_feat = jnp.tile(speed_feat, (self._params["n_rays"], 1))  # shape (n_rays,3)
            assert speed_feat.shape == (self._params["n_rays"], 3)
            # 构建特征 [dx_body, dy_body, u, v, r]
            lidar_feats = jnp.concatenate([
                pos_diff,  # [n_rays,2]
                speed_feat      # [n_rays,3]
            ], axis=1)  # -> [n_rays,5]
            assert lidar_feats.shape == (self._params["n_rays"], 5)
            # 激活掩码
            lidar_pos = agent_pos[i, :] - lidar_data[id_hits, :2]
            lidar_dist = jnp.linalg.norm(lidar_pos, axis=-1)
            active_lidar = jnp.less(lidar_dist, self._params["comm_radius"] - 1e-1)
            
            agent_obs_mask = jnp.ones((1, self._params["n_rays"]))
            agent_obs_mask = jnp.logical_and(agent_obs_mask, active_lidar)
            agent_obs_edges.append(
                EdgeBlock(lidar_feats[None, :, :], agent_obs_mask, id_agent[i][None], id_obs[id_hits])
            )
        
        assert len(agent_obs_edges) == self.num_agents
        return [agent_agent] + agent_goal + agent_obs_edges


    def control_affine_dyn(self, state: State) -> [Array, Array]:
        assert state.ndim == 2
        f = jnp.concatenate([
            (jnp.cos(state[:, 2]) * state[:, 3])[:, None],
            (jnp.sin(state[:, 2]) * state[:, 3])[:, None],
            jnp.zeros((state.shape[0], 2))
        ], axis=1)
        g = jnp.concatenate([jnp.zeros((2, 2)), jnp.array([[10., 0.], [0., 1.]])], axis=0)
        g = jnp.expand_dims(g, axis=0).repeat(f.shape[0], axis=0)
        assert f.shape == state.shape
        assert g.shape == (state.shape[0], 4, 2)
        return f, g

    def add_edge_feats(self, graph: GraphsTuple, state: State) -> GraphsTuple:
        assert graph.is_single
        assert state.ndim == 2  # state shape: [num_nodes, 6]

        # 获取发送端和接收端索引
        senders = graph.senders
        receivers = graph.receivers
        
        # ===== 本体坐标系转换 =====
        def body_relative(sender_state, receiver_pos):
            """将接收端位置转换到发送端本体坐标系"""
            phi = sender_state[2]
            R_inv = jnp.array([[jnp.cos(phi), jnp.sin(phi)],
                            [-jnp.sin(phi), jnp.cos(phi)]])
            return R_inv @ (receiver_pos - sender_state[:2])

        # 计算本体相对位置
        sender_states = state[senders]  # [num_edges, 6]
        receiver_pos = state[receivers, :2]  # [num_edges, 2]
        body_pos_diff = jax.vmap(body_related)(sender_states, receiver_pos)  # [num_edges, 2]

        # ===== 速度差计算 =====
        u_diff = sender_states[:, 3] - state[receivers, 3]  # du
        v_diff = sender_states[:, 4] - state[receivers, 4]  # dv
        r_diff = sender_states[:, 5] - state[receivers, 5]  # dr

        # ===== 构建边特征 =====
        edge_feats = jnp.concatenate([
            body_pos_diff,          # dx_body, dy_body
            u_diff[:, None],        # du
            v_diff[:, None],        # dv
            r_diff[:, None]         # dr
        ], axis=1)  # [num_edges, 5]

        # ===== 特征归一化 =====
        pos_norm = jnp.linalg.norm(body_pos_diff, axis=1, keepdims=True)
        comm_radius = self._params["comm_radius"]
        scale = jnp.where(pos_norm > comm_radius, comm_radius/pos_norm, 1.0)
        edge_feats = edge_feats.at[:, :2].set(edge_feats[:, :2] * scale)

        return graph._replace(edges=edge_feats, states=state)

    def get_graph(self, state: EnvState, adjacency: Array = None) -> GraphsTuple:
        # 节点特征（保持不变）
        n_hits = self._params["n_rays"] * self.num_agents
        n_nodes = 2 * self.num_agents + n_hits
        node_feats = jnp.zeros((n_nodes, 3))
        node_feats = node_feats.at[:self.num_agents, 2].set(1)    # Agent节点: [0,0,1]
        node_feats = node_feats.at[self.num_agents:self.num_agents*2, 1].set(1)  # Goal节点: [0,1,0]
        node_feats = node_feats.at[-n_hits:, 0].set(1)            # Obstacle节点: [1,0,0]

        # 节点类型标识
        node_type = jnp.zeros(n_nodes, dtype=jnp.int32)
        node_type = node_type.at[self.num_agents:self.num_agents*2].set(USV.GOAL)
        node_type = node_type.at[-n_hits:].set(USV.OBS)

        get_lidar_vmap = jax.vmap(
            ft.partial(
                get_lidar,
                obstacles=state.obstacle,
                num_beams=self._params["n_rays"],
                sense_range=self._params["comm_radius"],
            )
        )
        lidar_data = merge01(get_lidar_vmap(state.agent[:, :2]))
        lidar_data = jnp.concatenate([lidar_data, jnp.zeros((lidar_data.shape[0], 4))], axis=-1)
        edge_blocks = self.edge_blocks(state, lidar_data)

        # create graph
        return GetGraph(
            nodes=node_feats,
            node_type=node_type,
            edge_blocks=edge_blocks,
            env_states=state,
            states=jnp.concatenate([state.agent, state.goal, lidar_data], axis=0),
        ).to_padded()
    
    def state_lim(self, state: Optional[State] = None) -> Tuple[State, State]:
        """
        Returns
        -------
        lower_limit, upper_limit: Tuple[State, State],
            limits of the state
        """

        lower_lim = jnp.array([-jnp.inf, -jnp.inf, -jnp.pi, -5, -2, -3])
        upper_lim = jnp.array([jnp.inf, jnp.inf, jnp.pi, -5, -2, -3])
        return lower_lim, upper_lim

    def action_lim(self) -> Tuple[Action, Action]:
        """
        Returns
        -------
        lower_limit, upper_limit: Tuple[Action, Action],
            limits of the action
        """
        lower_lim = jnp.ones(2) * -3.0
        upper_lim = jnp.ones(2) * 3.0
        return lower_lim, upper_lim

    def u_ref(self, graph: GraphsTuple) -> Action:
        agent_states = graph.type_states(type_idx=0, n_type=self.num_agents)
        goal_states = graph.type_states(type_idx=1, n_type=self.num_agents)
        pos_diff = agent_states[:, :2] - goal_states[:, :2]

        # PID parameters
        k_omega = 1.0  # 0.5
        k_v = 2.3
        k_a = 2.5

        dist = jnp.linalg.norm(pos_diff, axis=-1)
        theta_t = jnp.arctan2(-pos_diff[:, 1], -pos_diff[:, 0]) % (2 * jnp.pi)
        theta = agent_states[:, 2] % (2 * jnp.pi)
        theta_diff = theta_t - theta
        omega = jnp.zeros(agent_states.shape[0])
        agent_dir = jnp.concatenate([jnp.cos(theta)[:, None], jnp.sin(theta)[:, None]], axis=-1)
        assert agent_dir.shape == (agent_states.shape[0], 2)
        theta_between = jnp.arccos(
            jnp.clip(jnp.matmul(-pos_diff[:, None, :], agent_dir[:, :, None]).squeeze() / (dist + 0.0001),
                     a_min=-1, a_max=1))

        # when theta <= pi
        # anti-clockwise
        omega = jnp.where(jnp.logical_and(jnp.logical_and(theta_diff < jnp.pi, theta_diff >= 0), theta <= jnp.pi),
                          k_omega * theta_between, omega)
        # clockwise
        omega = jnp.where(jnp.logical_and(
            jnp.logical_not(jnp.logical_and(theta_diff < jnp.pi, theta_diff >= 0)), theta <= jnp.pi),
            -k_omega * theta_between, omega
        )

        # when theta > pi
        # clockwise
        omega = jnp.where(jnp.logical_and(jnp.logical_and(theta_diff > -jnp.pi, theta_diff <= 0), theta > jnp.pi),
                          -k_omega * theta_between, omega)
        # anti-clockwise
        omega = jnp.where(jnp.logical_and(
            jnp.logical_not(jnp.logical_and(theta_diff > -jnp.pi, theta_diff <= 0)), theta > jnp.pi),
            k_omega * theta_between, omega
        )

        omega = jnp.clip(omega, a_min=-5., a_max=5.)

        pos_diff_norm = jnp.sqrt(1e-6 + jnp.sum(pos_diff ** 2, axis=-1, keepdims=True))
        comm_radius = self._params["comm_radius"]
        safe_feats_norm = jnp.maximum(pos_diff_norm, comm_radius)
        coef = jnp.where(pos_diff_norm > comm_radius, comm_radius / safe_feats_norm, 1.0)
        pos_diff = coef * pos_diff
        a = -k_a * agent_states[:, 3] + k_v * jnp.linalg.norm(pos_diff, axis=-1)

        action = jnp.concatenate([omega[:, None], a[:, None]], axis=-1)
        return action

    def forward_graph(self, graph: GraphsTuple, action: Action) -> GraphsTuple:
        # calculate next graph
        agent_states = graph.type_states(type_idx=0, n_type=self.num_agents)
        goal_states = graph.type_states(type_idx=1, n_type=self.num_agents)
        obs_states = graph.type_states(type_idx=2, n_type=self._params["n_rays"] * self.num_agents)
        action = self.clip_action(action)

        assert action.shape == (self.num_agents, self.action_dim)
        assert agent_states.shape == (self.num_agents, self.state_dim)

        stop_mask = self.stop_mask(graph)
        next_agent_states = self.agent_step_euler(agent_states, action, stop_mask)
        next_states = jnp.concatenate([next_agent_states, goal_states, obs_states], axis=0)

        next_graph = self.add_edge_feats(graph, next_states)
        ipdb.set_trace()
        return next_graph

    def safe_mask(self, graph: GraphsTuple) -> Array:
        agent_pos = graph.type_states(type_idx=0, n_type=self.num_agents)[:, :2]

        # agents are not colliding
        pos_diff = agent_pos[:, None, :] - agent_pos[None, :, :]  # [i, j]: i -> j
        dist = jnp.linalg.norm(pos_diff, axis=-1)
        dist = dist + jnp.eye(dist.shape[1]) * (self._params["car_radius"] * 2 + 1)  # remove self connection
        safe_agent = jnp.greater(dist, self._params["car_radius"] * 4)

        safe_agent = jnp.min(safe_agent, axis=1)

        safe_obs = jnp.logical_not(
            inside_obstacles(agent_pos, graph.env_states.obstacle, self._params["car_radius"] * 2)
        )

        safe_mask = jnp.logical_and(safe_agent, safe_obs)

        return safe_mask

    @ft.partial(jax.jit, static_argnums=(0,))
    def unsafe_mask(self, graph: GraphsTuple) -> Array:
        agent_state = graph.type_states(type_idx=0, n_type=self.num_agents)
        agent_pos = agent_state[:, :2]

        # agents are colliding
        agent_pos_diff = agent_pos[None, :, :] - agent_pos[:, None, :]
        agent_dist = jnp.linalg.norm(agent_pos_diff, axis=-1)
        agent_dist = agent_dist + jnp.eye(agent_dist.shape[1]) * (self._params["car_radius"] * 2 + 1)
        unsafe_agent = jnp.less(agent_dist, self._params["car_radius"] * 2)
        unsafe_agent = jnp.max(unsafe_agent, axis=1)

        # agents are colliding with obstacles
        unsafe_obs = inside_obstacles(agent_pos, graph.env_states.obstacle, self._params["car_radius"] * 1.5)

        collision_mask = jnp.logical_or(unsafe_agent, unsafe_obs)

        # unsafe direction
        agent_warn_dist = 3 * self._params["car_radius"]
        obs_warn_dist = 2 * self._params["car_radius"]
        obs_pos = graph.type_states(type_idx=2, n_type=self._params["n_rays"] * self.num_agents)[:, :2]
        obs_pos_diff = obs_pos[None, :, :] - agent_pos[:, None, :]
        obs_dist = jnp.linalg.norm(obs_pos_diff, axis=-1)
        pos_diff = jnp.concatenate([agent_pos_diff, obs_pos_diff], axis=1)
        warn_zone = jnp.concatenate([jnp.less(agent_dist, agent_warn_dist), jnp.less(obs_dist, obs_warn_dist)], axis=1)
        pos_vec = (pos_diff / (jnp.linalg.norm(pos_diff, axis=2, keepdims=True) + 0.0001))
        heading_vec = jnp.concatenate([jnp.cos(agent_state[:, 2])[:, None],
                                       jnp.sin(agent_state[:, 2])[:, None]], axis=1)[:, None, :]
        heading_vec = heading_vec.repeat(pos_vec.shape[1], axis=1)
        inner_prod = jnp.sum(pos_vec * heading_vec, axis=2)
        unsafe_theta_agent = jnp.arctan2(self._params['car_radius'] * 2,
                                         jnp.sqrt(agent_dist ** 2 - 4 * self._params['car_radius'] ** 2))
        unsafe_theta_obs = jnp.arctan2(self._params['car_radius'],
                                       jnp.sqrt(obs_dist ** 2 - self._params['car_radius'] ** 2))
        unsafe_theta = jnp.concatenate([unsafe_theta_agent, unsafe_theta_obs], axis=1)
        lidar_mask = jnp.ones((self._params["n_rays"],))
        lidar_mask = jax.scipy.linalg.block_diag(*[lidar_mask] * self.num_agents)
        valid_mask = jnp.concatenate([jnp.ones((self.num_agents, self.num_agents)), lidar_mask], axis=-1)
        warn_zone = jnp.logical_and(warn_zone, valid_mask)
        unsafe_dir = jnp.max(jnp.logical_and(warn_zone, jnp.greater(inner_prod, jnp.cos(unsafe_theta))), axis=1)

        return jnp.logical_or(collision_mask, unsafe_dir)

    def collision_mask(self, graph: GraphsTuple) -> Array:
        agent_pos = graph.type_states(type_idx=0, n_type=self.num_agents)[:, :2]

        # agents are colliding
        pos_diff = agent_pos[:, None, :] - agent_pos[None, :, :]  # [i, j]: i -> j
        dist = jnp.linalg.norm(pos_diff, axis=-1)
        dist = dist + jnp.eye(dist.shape[1]) * (self._params["car_radius"] * 2 + 1)  # remove self connection
        unsafe_agent = jnp.less(dist, self._params["car_radius"] * 2)
        unsafe_agent = jnp.max(unsafe_agent, axis=1)

        # agents are colliding with obstacles
        unsafe_obs = inside_obstacles(agent_pos, graph.env_states.obstacle, self._params["car_radius"])

        collision_mask = jnp.logical_or(unsafe_agent, unsafe_obs)

        return collision_mask

    def finish_mask(self, graph: GraphsTuple) -> Array:
        agent_pos = graph.type_states(type_idx=0, n_type=self.num_agents)[:, :2]
        goal_pos = graph.env_states.goal[:, :2]
        reach = jnp.linalg.norm(agent_pos - goal_pos, axis=1) < self._params["car_radius"] * 2
        return reach

    def stop_mask(self, graph: GraphsTuple) -> Array:
        agent_pos = graph.type_states(type_idx=0, n_type=self.num_agents)[:, :2]
        goal_pos = graph.env_states.goal[:, :2]
        stop = jnp.linalg.norm(agent_pos - goal_pos, axis=1) < self._params["car_radius"] * 0.5
        return stop
