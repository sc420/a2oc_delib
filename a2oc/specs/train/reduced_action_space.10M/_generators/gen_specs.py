from a2oc.specs.train._generators.common import gen_specs

# Set the settings
settings = {
    'name': 'reduced_action_space.10M',
    'env_ids': [
        'BeamRiderNoFrameskip-v4',
        'KungFuMasterNoFrameskip-v4',
        'QbertNoFrameskip-v4',
        'SeaquestNoFrameskip-v4',
    ],
    'template_path_to_read': 'a2oc/specs/train/{name}/_generators/template.yml',
    'spec_path_to_write': 'a2oc/specs/train/{name}/env_id-{env_id}.yml',
}

if __name__ == '__main__':
    gen_specs(settings)
