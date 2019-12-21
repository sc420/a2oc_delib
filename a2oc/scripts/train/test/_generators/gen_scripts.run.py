from a2oc.scripts.train._generators.common import gen_scripts

# Set the settings
settings = {
    'name': 'test',
    'env_ids': [
        'BeamRiderNoFrameskip-v4',
    ],
    'script_path_to_write': 'a2oc/scripts/train/{name}/run.env_id-{env_id}.seed-{random_seed}.sh',
}

if __name__ == '__main__':
    gen_scripts(settings)
