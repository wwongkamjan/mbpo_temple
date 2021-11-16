import numpy as np


class PredictEnv:
    def __init__(self, model, env_name, model_type, args):
        self.model = model
        self.env_name = env_name
        self.model_type = model_type
        self.args = args

    def _termination_fn(self, env_name, obs, act, next_obs):
        # TODO
        if env_name == "Hopper-v2":
            assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2

            height = next_obs[:, 0]
            angle = next_obs[:, 1]
            not_done = np.isfinite(next_obs).all(axis=-1) \
                       * np.abs(next_obs[:, 1:] < 100).all(axis=-1) \
                       * (height > .7) \
                       * (np.abs(angle) < .2)

            done = ~not_done
            done = done[:, None]
            return done
        elif env_name == "Walker2d-v2":
            assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2

            height = next_obs[:, 0]
            angle = next_obs[:, 1]
            not_done = (height > 0.8) \
                       * (height < 2.0) \
                       * (angle > -1.0) \
                       * (angle < 1.0)
            done = ~not_done
            done = done[:, None]
            return done
        elif env_name == "HalfCheetah-v2":
            assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2

            done = np.array([False]).repeat(len(obs))
            done = done[:, None]
            return done
        elif env_name == "Ant-v2":
            assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2

            x = next_obs[:, 0]
            not_done = np.isfinite(next_obs).all(axis=-1) \
                       * (x >= 0.2) \
                       * (x <= 1.0)

            done = ~not_done
            done = done[:, None]
            return done
        elif env_name == "Humanoid-v2":
            assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2

            z = next_obs[:, 0]
            done = (z < 1.0) + (z > 2.0)

            done = done[:, None]
            return done
        elif env_name == "Swimmer-v2":
            assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2

            done = np.array([False]).repeat(len(obs))
            done = done[:, None]
            return done
        elif env_name == "Striker-v2":
            assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2

            done = np.array([False]).repeat(len(obs))
            done = done[:, None]
            return done
        elif env_name == "Thrower-v2":
            assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2

            done = np.array([False]).repeat(len(obs))
            done = done[:, None]
            return done
        elif env_name == "Pusher-v2":
            assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2

            done = np.array([False]).repeat(len(obs))
            done = done[:, None]
            return done
        elif env_name == "Reacher-v2":
            assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2

            done = np.array([False]).repeat(len(obs))
            done = done[:, None]
            return done
        elif 'walker_' in env_name:
            torso_height = next_obs[:, -2]
            torso_ang = next_obs[:, -1]
            if 'walker_7' in env_name or 'walker_5' in env_name:
                offset = 0.
            else:
                offset = 0.26
            not_done = (torso_height > 0.8 - offset) \
                       * (torso_height < 2.0 - offset) \
                       * (torso_ang > -1.0) \
                       * (torso_ang < 1.0)
            done = ~not_done
            done = done[:, None]
            return done

    def _get_logprob(self, x, means, variances):

        k = x.shape[-1]

        ## [ num_networks, batch_size ]
        log_prob = -1 / 2 * (k * np.log(2 * np.pi) + np.log(variances).sum(-1) + (np.power(x - means, 2) / variances).sum(-1))

        ## [ batch_size ]
        prob = np.exp(log_prob).sum(0)

        ## [ batch_size ]
        log_prob = np.log(prob)

        stds = np.std(means, 0).mean(-1)

        return log_prob, stds

    def step(self, obs, act, deterministic=False):
        if len(obs.shape) == 1:
            obs = obs[None]
            act = act[None]
            return_single = True
        else:
            return_single = False

        inputs = np.concatenate((obs, act), axis=-1)
        if self.model_type == 'pytorch':
            ensemble_model_means, ensemble_model_vars = self.model.predict(inputs)
        else:
            ensemble_model_means, ensemble_model_vars = self.model.predict(inputs, factored=True)
        ensemble_model_means[:, :, 1:] += obs
        ensemble_model_stds = np.sqrt(ensemble_model_vars)

        if deterministic:
            ensemble_samples = ensemble_model_means
        else:
            ensemble_samples = ensemble_model_means + np.random.normal(size=ensemble_model_means.shape) * ensemble_model_stds

        num_models, batch_size, _ = ensemble_model_means.shape
        if self.model_type == 'pytorch':
            model_idxes = np.random.choice(self.model.elite_model_idxes, size=batch_size)
        else:
            model_idxes = self.model.random_inds(batch_size)
        batch_idxes = np.arange(0, batch_size)

        samples = ensemble_samples[model_idxes, batch_idxes]
        model_means = ensemble_model_means[model_idxes, batch_idxes]
        model_stds = ensemble_model_stds[model_idxes, batch_idxes]


        log_prob, dev = self._get_logprob(samples, ensemble_model_means, ensemble_model_vars)

        rewards, next_obs = samples[:, :1], samples[:, 1:]
        terminals = self._termination_fn(self.env_name, obs, act, next_obs)

        if self.args.exploration:
            model_index = np.arange(0, num_models).tolist()
            model_idxes_rest = np.array([model_index[:i] + model_index[i + 1:] for i in model_idxes])
            model_vars = ensemble_model_vars[model_idxes, batch_idxes]
            model_means_rest = np.array(
                [ensemble_model_means[model_idxes_rest[:, n], batch_idxes, 1:] for n in range(num_models - 1)])
            model_vars_rest = np.array(
                [ensemble_model_vars[model_idxes_rest[:, n], batch_idxes, 1:] for n in range(num_models - 1)])
            model_means_rest = np.mean(model_means_rest, axis=0)
            model_vars_rest = np.mean(model_vars_rest + model_means_rest ** 2, axis=0) - model_means_rest ** 2
            KLdivergence_list = np.log(model_vars_rest / model_vars[:, 1:]) - 0.5 + (
                    model_vars[:, 1:] + np.square(model_means[:, 1:] - model_means_rest)) / (2 * model_vars_rest)
            KL = np.sum(KLdivergence_list, axis=-1)
            gamma = np.exp((-1) * KL)
            # sample_chosen_inds = np.argsort(KL_result)[:int(mask_rate * batch_size)]
            # terminals_rollout = np.array([[False] if i in sample_chosen_inds else [True] for i in batch_inds])
            KL_result = np.reshape(KL, [batch_size, 1])
            rewards += 1 * KL_result

        batch_size = model_means.shape[0]
        return_means = np.concatenate((model_means[:, :1], terminals, model_means[:, 1:]), axis=-1)
        return_stds = np.concatenate((model_stds[:, :1], np.zeros((batch_size, 1)), model_stds[:, 1:]), axis=-1)

        if return_single:
            next_obs = next_obs[0]
            return_means = return_means[0]
            return_stds = return_stds[0]
            rewards = rewards[0]
            terminals = terminals[0]

        info = {'mean': return_means, 'std': return_stds, 'log_prob': log_prob, 'dev': dev}
        return next_obs, rewards, terminals, info
