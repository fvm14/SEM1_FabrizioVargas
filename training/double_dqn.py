import numpy as np
import torch as th
from torch.nn import functional as F
from stable_baselines3 import DQN

class DoubleDQN(DQN):
    """
    Double Deep Q-Network (Van Hasselt et al., 2016).

    Stable-Baselines3 implementa unicamente DQN vanilla (Mnih et al., 2015):
    el target se calcula como max_a Q_target(s', a), donde la misma red
    objetivo selecciona y evalua la accion, lo que produce sobreestimacion
    sistematica de los valores Q.

    Esta subclase sobreescribe el metodo train() para aplicar la correccion
    de Double DQN: la red online selecciona la accion del siguiente estado
    (argmax_a Q_online(s', a)) y la red objetivo la evalua
    (Q_target(s', argmax_a Q_online(s', a))). Al desacoplar seleccion y
    evaluacion, ninguna red infla sus propias estimaciones para justificar
    sus propias decisiones.

    El resto del algoritmo (replay buffer, exploracion epsilon-greedy,
    actualizacion de la red objetivo, perdida Huber) se hereda sin cambios
    de la implementacion de DQN en SB3, de modo que la unica diferencia
    entre DQN y DDQN en este trabajo es la regla de calculo del target.
    """

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)

        losses = []
        for _ in range(gradient_steps):
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)

            with th.no_grad():

                next_actions = self.q_net(replay_data.next_observations).argmax(dim=1, keepdim=True)

                next_q_values = th.gather(
                    self.q_net_target(replay_data.next_observations),
                    dim=1,
                    index=next_actions
                )

                target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values

            current_q_values = th.gather(
                self.q_net(replay_data.observations),
                dim=1,
                index=replay_data.actions.long()
            )

            loss = F.smooth_l1_loss(current_q_values, target_q_values)
            losses.append(loss.item())

            self.policy.optimizer.zero_grad()
            loss.backward()
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

        self._n_updates += gradient_steps
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/loss", np.mean(losses))
