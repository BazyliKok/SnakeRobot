from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read_source(relative_path):
    return (ROOT / relative_path).read_text(encoding="utf-8")


def test_separate_terrain_mode_is_the_default():
    train_source = read_source("CoadaptationCode/train_coadapt.py")
    sac_source = read_source("CoadaptationCode/soft_actor_critic_coadapt.py")
    replay_source = read_source("CoadaptationCode/replaybuffercoadapt.py")

    assert "SNAKE_TERRAIN_MODEL_MODE', 'separate'" in train_source
    assert "SNAKE_TERRAIN_MODEL_MODE', 'separate'" in sac_source
    assert "terrain_model_mode='separate'" in replay_source
    assert "SNAKE_BOOTSTRAP_INDIVIDUAL_FROM_TERRAIN_POPULATION" in train_source
    assert "SNAKE_RESUME_COMPLETED_TERRAIN_BLOCKS" in train_source
    assert "SNAKE_FIXED_SCALE_DESIGN" in train_source
    assert "SNAKE_RESUME_AS_NEW_DESIGN_RUN" in train_source
    assert "SNAKE_CHECKPOINT_RESULTS_TAG" in train_source


def test_fresh_separate_individual_bootstraps_from_same_terrain_population():
    train_source = read_source("CoadaptationCode/train_coadapt.py")

    separate_branch_start = train_source.index("Initializing fresh individual policy for terrain")
    separate_branch = train_source[separate_branch_start:train_source.index("self._last_individual_reset_key", separate_branch_start)]

    assert "self.bootstrap_individual_from_terrain_population" in separate_branch
    assert "copy_population_to_individual=(" in separate_branch
    assert "reset_individual=reset_fresh" in separate_branch
    assert "Selecting existing individual policy for terrain" in separate_branch
    assert "not copying from population" in separate_branch


def test_separate_replay_species_sampling_uses_same_terrain_individual_population_split():
    replay_source = read_source("CoadaptationCode/replaybuffercoadapt.py")

    random_batch_start = replay_source.index("def random_batch")
    species_start = replay_source.index('if self._mode == "species":', random_batch_start)
    species_block = replay_source[species_start:replay_source.index('elif self._mode == "population":', species_start)]

    assert "_individual_batch_fraction" in species_block
    assert "self._balanced_random_batch(self._population_buffer, pop_batch_size)" in species_block
    assert "self._balanced_random_batch(self._individual_buffer, ind_batch_size)" in species_block


def test_resume_can_skip_completed_requested_terrain_blocks():
    train_source = read_source("CoadaptationCode/train_coadapt.py")

    assert "def _completed_requested_terrain_blocks_from_checkpoint" in train_source
    assert "completed_blocks * requested_training_block_size" in train_source
    assert "Treating completed checkpoint terrain block(s) as done" in train_source


def test_pso_scores_with_terrain_specific_actor_and_critic_maps():
    pso_source = read_source("CoadaptationCode/pso_batch.py")

    assert "_network_for_terrain(q_network, terrain_name)" in pso_source
    assert "_network_for_terrain(policy_network, terrain_name)" in pso_source
    assert "terrain_q_network(network_input, action)" in pso_source
