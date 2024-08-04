# First, we need to understand how do we add owr own custom command.
# Second and third, we can read one file, and multiple ones and walk all of them and parser it
# Fourth is to just make mulirun work, and then we can add the other features.

# Necessary Imports
import os
import glob
import argparse
import itertools

class YAMLParser:
    ''' This class is used to parse the YAML file and return the dictionary. Doing other fancy stuff maybe later. '''
    def __init__(self, file_path):
        self.file_path = file_path

    def parse(self):
        import yaml
        with open(self.file_path) as f:
            # before returning, we need to add all groupname and properties as the property of self.
            
            data = yaml.load(f, Loader=yaml.FullLoader)
            # print(data)
            # Task 1: Make yaml.load as self.data
            setattr(self, "data", data)
            
            # Task2: Go through each keys, and set it as the property of the class.
            for key, value in data.items():
                print("Setting the property : ", key, value)
                setattr(self, key, value)
                for prop, val in value.items():
                    print("Setting the property : ", prop, val)
                    setattr(self, prop, val)

            return self

class CrackedDict(dict):
    def __setitem__(self, key, value) -> None:
        setattr(self, key, value)
        return super().__setitem__(key, value)
    
    def __repr__(self) -> str:
        return f"Cracked({super().__repr__()})"
     

class RunTest:
    """Testing class for the multirun functionality. """
    def __init__(self):
        pass 

    def run(self, config_dir):
        def decorator(func):
            def wrapper(*args, **kwargs):
                
                print("------ Running the test --------- ")

                # The problem is that the in between the final file and the cwd, there could be multiple folders. How to navigate this?
                #print(os.path.abspath(__file__)) # if we remove the file, we get the current directory

                # We need all of these, but for starting, some are doing the job well. We'll get fancy later.
                dir_path = os.path.dirname(os.path.abspath(__file__))
                
                # Assumption : Make sure the base config folder is present.
                if not os.path.exists(os.path.join(dir_path, config_dir)):
                    raise FileNotFoundError(f"Config folder {config_dir} not found in the directory {dir_path}")
                
                config_folder = os.path.join(dir_path, config_dir)
                all_folder = [folder for folder in glob.glob(f"{config_folder}/*") if os.path.isdir(folder)]

                # Assumption : Make sure atleast 1 group should be there
                if not all_folder:
                    raise FileNotFoundError(f"No group found in the config folder {config_dir}")
                
                group_names = [os.path.basename(folder) for folder in all_folder]
                
                _yaml = CrackedDict()
                config_store = []

                # The routine is simple. Walk through all the folders, and then get all the yaml files in them. Prse those in a dictionary.
                for group,folder in zip(group_names, all_folder): # They are always the same length.
                    
                    # folder_name = os.path.basename(folder) # This is the group name
                    _yaml[group] = CrackedDict()
                    all_yaml_files = glob.glob(f"{folder}/*.yaml")  # This is the list of all the yaml files in the folder.
                    
                    for yaml_file in all_yaml_files:
                        
                        base = os.path.basename(yaml_file)
                        file_name = os.path.splitext(base)[0] # This is the file name(no extenstion)
                        
                        obj = YAMLParser(yaml_file)
                        cfg = obj.parse()
                        _yaml[group][file_name] = cfg  # This is the dictionary of the file.                        
                        config_store.append(cfg)  # This is the dictionary of the file.
                    

                parser = argparse.ArgumentParser()
                parser.add_argument("--multirun", type=str, default="", help="Multirun options", nargs="+") # nargs+ means multiple arguments.
                cli_args = parser.parse_args()

                if cli_args.multirun:
                    multirun_options = self.parse_multirun(cli_args.multirun)  # This is the list of the commands, all or without the combinations to be used.

                    combinations = self.generate_combinations(multirun_options) # All the combinations of the multirun options.
                    print("==================================================================")
                    fields = ["numbers_group_1"]
                    print(fields)
                    for field, combination in zip(fields, combinations):
                        print(f"[INFO] ======== Running for: {combination} ========== ")  # Some form of logging, could store in some random file.
                        for group, option in combination.items():
                            print(f"[INFO] Running for group: {group}, option: {option}")
                            self.apply_combination(_yaml, group, option, field, lambda x: x)
                        # self.apply_combinations(_yaml, combination, lambda x: x) # As of now, this function doesn't make too much sense.
                    print("==================================================================")
                        
                return func(_yaml, *args, **kwargs)
            return wrapper
        return decorator
    
    def parse_multirun(self, multirun_list):
        options = {}
        groups = multirun_list
        for group in groups:
            key, value = group.split('=')
            options[key] = value.split(',')
        # Currently, option is a dictionary of the group and the options. We need to add maybe a property field as well.
         
        return options

    def generate_combinations(self, options):
        print("==================================================================")
        # print("Options: ",options)
        keys = options.keys()
        values = options.values()
        return [dict(zip(keys, combination)) for combination in itertools.product(*values)]
        print("==================================================================")

    def apply_combinations(self, config, combination, func):
        # On the surface, this function doesn't make too much sense. We can, for example give a function we can apply to it.
        # We can also give a function that can be applied to it.
        pass
        
    def apply_combination(self, config, group, option, field, func):
        print("==================================================================")
        val = getattr(config[group][option], field)
        print(f"Applying the function to the config: {val}")
        return func(val)
        print("==================================================================")
        

obj = RunTest()

@obj.run(config_dir = "base")
def main(cfg):
    print(f"This is the config : {cfg}")

if __name__ == "__main__":
    main()

# Great job for the first prototype. I'll make a problem statement and then we can refine it.