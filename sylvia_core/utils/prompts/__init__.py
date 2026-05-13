import importlib
import logging
from pathlib import Path

from langchain_core.prompts import PromptTemplate

logger = logging.getLogger(__name__)


class PromptManager:
    def __init__(self, prompt_package_name="sylvia_core.prompts"):
        self.prompts = {}
        self.prompt_package_name = prompt_package_name
        self._load_prompts()

    def _load_prompts(self):
        try:
            prompts_base_module = importlib.import_module(self.prompt_package_name)

            package_path = None
            if hasattr(prompts_base_module, "__file__") and prompts_base_module.__file__:
                package_path = Path(prompts_base_module.__file__).parent
            elif hasattr(prompts_base_module, "__path__"):
                for p in prompts_base_module.__path__:
                    package_path = Path(p)
                    break

            if not package_path or not package_path.is_dir():
                raise ModuleNotFoundError(
                    f"Could not determine physical path for prompts package: {self.prompt_package_name}"
                )

            # Collect base prompt names (without .private or _example suffix)
            prompt_bases = set()
            for item in package_path.iterdir():
                if item.is_file() and item.name.endswith(".py") and item.name != "__init__.py":
                    stem = item.stem
                    if stem.endswith(".private"):
                        prompt_bases.add(stem[:-8])  # Remove ".private"
                    elif stem.endswith("_example"):
                        prompt_bases.add(stem[:-8])  # Remove "_example"
                    elif not stem.startswith("_") and stem.isidentifier():
                        prompt_bases.add(stem)

            # Load each prompt: try .private first, fallback to _example
            for base_name in prompt_bases:
                loaded = False
                for suffix in [".private", "_example", ""]:
                    module_name = f"{base_name}{suffix}"
                    full_module_name = f"{self.prompt_package_name}.{module_name}"
                    try:
                        prompt_module = importlib.import_module(full_module_name)
                        if hasattr(prompt_module, "PROMPT"):
                            self.prompts[base_name] = getattr(prompt_module, "PROMPT")
                            loaded = True
                            break
                    except (ImportError, ModuleNotFoundError):
                        continue
                    except Exception:
                        logger.exception("Failed to load prompt module: %s", full_module_name)
                        break

                if not loaded:
                    logger.warning("Could not load prompt: %s (tried .private, _example, base)", base_name)

        except ModuleNotFoundError:
            logger.exception("Prompts package not found: %s", self.prompt_package_name)
        except Exception:
            logger.exception("Unexpected error while loading prompts")

    def get(self, prompt_name: str) -> str:
        return self.prompts.get(prompt_name, "")

    def get_prompt_template(self, prompt_name: str) -> PromptTemplate:
        template_string = self.get(prompt_name)
        if not template_string:
            raise ValueError(f"Prompt '{prompt_name}' not found.")
        return PromptTemplate.from_template(template_string)


prompt_manager = PromptManager()
