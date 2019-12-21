from a2oc._generators.common import read_file, write_file

# Set the random seeds
RANDOM_SEEDS = range(1000, 1000 + 5)

# Set the default common template content to read
DEFAULT_COMMON_TEMPLATE_CONTENT_PATH_TO_READ = \
    'a2oc/specs/train/_generators/common.template.yml'

# Set the default run template content to read
DEFAULT_RUN_TEMPLATE_CONTENT_PATH_TO_READ = \
    'a2oc/specs/train/_generators/run.template.yml'

# Set the default download template content to read
DEFAULT_DOWNLOAD_TEMPLATE_CONTENT_PATH_TO_READ = \
    'a2oc/specs/train/_generators/download.template.yml'


def gen_specs(settings):
    # Read the spec template
    spec_template = read_template(settings, 'template_path_to_read')

    # Read the spec template for common content
    common_template = read_template(
        settings, 'common_template_content_path_to_read',
        DEFAULT_COMMON_TEMPLATE_CONTENT_PATH_TO_READ)

    # Read the spec template for run contents
    run_template = read_template(
        settings, 'run_template_content_path_to_read',
        DEFAULT_RUN_TEMPLATE_CONTENT_PATH_TO_READ)

    # Read the spec template for download contents
    download_template = read_template(
        settings, 'download_template_content_path_to_read',
        DEFAULT_DOWNLOAD_TEMPLATE_CONTENT_PATH_TO_READ)

    # Write the specs
    write_specs(settings, spec_template, common_template,
                run_template, download_template)


def read_template(settings, key, default_path=None):
    # Get the template path
    template_path = settings.get(key, None)

    if template_path is None:
        template_path = default_path
    else:
        # Fill the name
        template_path = template_path.format(name=settings['name'])

    # Read the file and return the content
    return read_file(template_path)


def write_specs(
        settings, spec_template, common_template, run_template,
        download_template):
    # Write all specs
    for format_mapping in gen_spec_format_mappings(settings):
        # Build the common content
        commom_content = build_common_content(common_template, format_mapping)

        # Build the run contents
        run_contents = build_content_parts(run_template)

        # Build the download contents
        download_contents = build_content_parts(download_template)

        # Add the contents to the format mapping
        format_mapping['common_content'] = commom_content
        format_mapping['run_contents'] = run_contents
        format_mapping['download_contents'] = download_contents

        # Build the spec path
        spec_path = settings['spec_path_to_write'].format(**format_mapping)

        # Build the spec content
        spec_content = spec_template.format(**format_mapping)

        # Write the spec content
        write_file(spec_path, spec_content)


def build_common_content(common_template, format_mapping):
    # Build the common content and return
    return common_template.format(**format_mapping)


def build_content_parts(template):
    # Initialize the parts
    parts = []

    # Build all content parts
    for format_mapping in gen_parts_format_mappings():
        # Build the content part
        part = template.format(**format_mapping)

        # Add to the parts
        parts.append(part)

    # Merge all parts and return
    return '\n'.join(parts)


def gen_spec_format_mappings(settings):
    # Generate for all environments
    for env_id in settings['env_ids']:
        # Build the format mapping and yield
        yield {
            'name': settings['name'],
            'env_id': env_id,
        }


def gen_parts_format_mappings():
    # Generate for all random seeds
    for random_seed in RANDOM_SEEDS:
        # Build the format mapping and yield
        yield {
            'random_seed': random_seed,
        }
