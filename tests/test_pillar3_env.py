"""Test Pillar 3 env step and reward."""

from fedstable.pillar3_fedrl.env import StablecoinDeFiEnv


def test_env_reset_step():
    env = StablecoinDeFiEnv(state_dim=4, action_dim=2, max_steps=5, seed=42)
    state = env.reset()
    assert state.shape == (4,)
    state2, reward, done, info = env.step(0)
    assert state2.shape == (4,)
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    assert not done
    for _ in range(4):
        _, _, done, _ = env.step(0)
    assert done
