import yaml

def load(name, keys):
    if name == "placeholder":
        with open("./config/%s.yaml" % name, 'w') as f:
            print(keys)
            for key in keys:
                f.write("%s:\n" % key)
        print("created placeholder yaml")
        exit()

    with open("./config/%s.yaml" % name, 'r') as f:
        config = yaml.load(f)
        for key in keys:
            if key != "config_file":
                if key not in config:
                    raise ValueError("Config file missing param " + key)
    return config

