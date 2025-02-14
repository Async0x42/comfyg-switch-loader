import comfy.samplers
import folder_paths
import json
import os

class ComfygSwitchLoader:
    @classmethod
    def load_configs(cls):
        """
        Load configuration options from a JSON file and return them as a dictionary.
        """
        if not hasattr(cls, '_configs'):
            config_path = os.path.join(os.path.dirname(__file__), "model_configs.json")
            try:
                with open(config_path, "r") as f:
                    cls._configs = json.load(f)
            except Exception as e:
                print('ComfygSwitchLoader -> Exception loading configs:', e)
                cls._configs = {}
        return cls._configs

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # Input is now a checkpoint name string.
                "ckpt_name": (sorted(folder_paths.get_filename_list("checkpoints")), {"tooltip": "Name of the checkpoint to load."}),
                "use_custom_input": ("BOOLEAN", {"default": False}),
                "steps": ("INT", {"default": 30, "min": 1, "max": 200}),
                "cfg": ("FLOAT", {"default": 7.0, "min": 0.1, "max": 20.0, "step": 0.1}),
                "sampler": (comfy.samplers.KSampler.SAMPLERS, ),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, ),
            }
        }

    # Now outputs MODEL, CLIP, VAE plus configuration parameters.
    RETURN_TYPES = (
        "MODEL",
        "CLIP",
        "VAE",
        "INT",
        "FLOAT",
        comfy.samplers.KSampler.SAMPLERS,
        comfy.samplers.KSampler.SCHEDULERS
    )
    RETURN_NAMES = ("MODEL", "CLIP", "VAE", "STEPS", "CFG", "SAMPLER", "SCHEDULER")
    FUNCTION = "select_config"
    CATEGORY = "Configuration"

    def select_config(self, ckpt_name, use_custom_input, steps, cfg, sampler, scheduler):
        # Build the checkpoint file path.
        ckpt_path = folder_paths.get_full_path_or_raise("checkpoints", ckpt_name)
        try:
            # Load checkpoint with VAE and CLIP.
            out = comfy.sd.load_checkpoint_guess_config(
                ckpt_path,
                output_vae=True,
                output_clip=True,
                embedding_directory=folder_paths.get_folder_paths("embeddings")
            )
        except Exception as e:
            print("ComfygSwitchLoader -> Error loading checkpoint:", e)
            # In case of error, return None for model, clip, and vae and fallback to provided parameters.
            return (None, None, None, steps, cfg, sampler, scheduler)
        
        model, clip, vae = out[:3]
        # Attach a checkpoint name (without extension) to the model.
        model_name = os.path.splitext(ckpt_name)[0]
        setattr(model, "ckpt_name", model_name)
        print('ComfygSwitchLoader -> Loaded model:', model_name)

        # Build a configuration dictionary from the current node inputs.
        node_config = {
            "steps": steps,
            "cfg": cfg,
            "sampler": sampler,
            "scheduler": scheduler
        }

        # Load existing configurations.
        configs = self.load_configs()
        config_path = os.path.join(os.path.dirname(__file__), "model_configs.json")
        
        if not use_custom_input:
            stored_config = configs.get(model_name)
            # Update the JSON file if there is no stored config or if the inputs differ.
            if stored_config != node_config:
                configs[model_name] = node_config
                try:
                    with open(config_path, "w") as f:
                        json.dump(configs, f, indent=4)
                    print("ComfygSwitchLoader -> Updated config for", model_name)
                except Exception as e:
                    print("ComfygSwitchLoader -> Error writing config for", model_name, e)
            final_config = configs[model_name]
        else:
            final_config = node_config

        print('ComfygSwitchLoader -> final_config', final_config)

        if use_custom_input:
            return (model, clip, vae, steps, cfg, sampler, scheduler)
        else:
            return (
                model,
                clip,
                vae,
                final_config["steps"],
                final_config["cfg"],
                final_config["sampler"],
                final_config["scheduler"]
            )

NODE_CLASS_MAPPINGS = {
    "ComfygSwitchLoader": ComfygSwitchLoader
}
