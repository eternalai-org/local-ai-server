from yaml import load, Loader
import os
import pkg_resources

# Get the package's installation directory
package_dir = os.path.dirname(pkg_resources.resource_filename(__name__, '__init__.py'))
config_path = os.path.join(package_dir, '..', 'configs', '8x4090.yaml')

CONFIG = load(open(config_path, "r"), Loader=Loader)