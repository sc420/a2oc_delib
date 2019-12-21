from a2oc.scripts.noodles._generators.common import gen_scripts

# Set the settings
settings = {
    'name': 'test',
    'env_ids': [
        'BeamRiderNoFrameskip-v4'
    ],
    'script_path_to_write': 'a2oc/scripts/noodles/{name}/{noodles_action}.env_id-{env_id}.sh',
}

if __name__ == '__main__':
    gen_scripts(settings)
