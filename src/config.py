import sys
import os

def configure_pythonpath(config_file):
  """Configures the Python path based on the specified configuration file.

  Args:
    config_file: The path to the configuration file.
  """
  dirs = []
  with open(config_file, 'r') as f:
    for line in f:
      line = line.strip() # removes leading and trailing whitespaces
      if line and not line.startswith('#'):  # Ignore empty lines and lines with a leading '#'
        dirs.append(line)
  dirs = ';'.join(dirs)
  #os.environ['PYTHONPATH'] = dirs
  return dirs
  
if __name__ == "__main__":
  config_file = "./src/.paths.conf"
  print(configure_pythonpath(config_file))