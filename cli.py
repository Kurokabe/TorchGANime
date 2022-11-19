from typing import Any, Callable, Dict, List, Optional, Type, Union

from pytorch_lightning import (
    Callback,
    LightningDataModule,
    LightningModule,
    Trainer,
)
from pytorch_lightning.cli import (
    LightningCLI,
)

import ray.tune as tune

from pytorch_lightning.cli import SaveConfigCallback, ArgsType


class RayTuneCli(LightningCLI):
    def __init__(
        self,
        model_class: Optional[
            Union[Type[LightningModule], Callable[..., LightningModule]]
        ] = None,
        datamodule_class: Optional[
            Union[Type[LightningDataModule], Callable[..., LightningDataModule]]
        ] = None,
        save_config_callback: Optional[Type[SaveConfigCallback]] = SaveConfigCallback,
        save_config_kwargs: Optional[Dict[str, Any]] = None,
        trainer_class: Union[Type[Trainer], Callable[..., Trainer]] = Trainer,
        trainer_defaults: Optional[Dict[str, Any]] = None,
        seed_everything_default: Union[bool, int] = True,
        description: str = "pytorch-lightning trainer command line tool",
        env_prefix: str = "PL",
        env_parse: bool = False,
        parser_kwargs: Optional[
            Union[Dict[str, Any], Dict[str, Dict[str, Any]]]
        ] = None,
        subclass_mode_model: bool = False,
        subclass_mode_data: bool = False,
        args: ArgsType = None,
        run: bool = True,
        auto_registry: bool = False,
        **kwargs: Any,  # Remove with deprecations of v1.10
    ) -> None:
        """Receives as input pytorch-lightning classes (or callables which return pytorch-lightning classes), which
        are called / instantiated using a parsed configuration file and / or command line args.

        Parsing of configuration from environment variables can be enabled by setting ``env_parse=True``.
        A full configuration yaml would be parsed from ``PL_CONFIG`` if set.
        Individual settings are so parsed from variables named for example ``PL_TRAINER__MAX_EPOCHS``.

        For more info, read :ref:`the CLI docs <lightning-cli>`.

        .. warning:: ``LightningCLI`` is in beta and subject to change.

        Args:
            model_class: An optional :class:`~pytorch_lightning.core.module.LightningModule` class to train on or a
                callable which returns a :class:`~pytorch_lightning.core.module.LightningModule` instance when
                called. If ``None``, you can pass a registered model with ``--model=MyModel``.
            datamodule_class: An optional :class:`~pytorch_lightning.core.datamodule.LightningDataModule` class or a
                callable which returns a :class:`~pytorch_lightning.core.datamodule.LightningDataModule` instance when
                called. If ``None``, you can pass a registered datamodule with ``--data=MyDataModule``.
            save_config_callback: A callback class to save the config.
            save_config_kwargs: Parameters that will be used to instantiate the save_config_callback.
            trainer_class: An optional subclass of the :class:`~pytorch_lightning.trainer.trainer.Trainer` class or a
                callable which returns a :class:`~pytorch_lightning.trainer.trainer.Trainer` instance when called.
            trainer_defaults: Set to override Trainer defaults or add persistent callbacks. The callbacks added through
                this argument will not be configurable from a configuration file and will always be present for
                this particular CLI. Alternatively, configurable callbacks can be added as explained in
                :ref:`the CLI docs <lightning-cli>`.
            seed_everything_default: Number for the :func:`~lightning_lite.utilities.seed.seed_everything`
                seed value. Set to True to automatically choose a seed value.
                Setting it to False will avoid calling ``seed_everything``.
            description: Description of the tool shown when running ``--help``.
            env_prefix: Prefix for environment variables.
            env_parse: Whether environment variable parsing is enabled.
            parser_kwargs: Additional arguments to instantiate each ``LightningArgumentParser``.
            subclass_mode_model: Whether model can be any `subclass
                <https://jsonargparse.readthedocs.io/en/stable/#class-type-and-sub-classes>`_
                of the given class.
            subclass_mode_data: Whether datamodule can be any `subclass
                <https://jsonargparse.readthedocs.io/en/stable/#class-type-and-sub-classes>`_
                of the given class.
            args: Arguments to parse. If ``None`` the arguments are taken from ``sys.argv``. Command line style
                arguments can be given in a ``list``. Alternatively, structured config options can be given in a
                ``dict`` or ``jsonargparse.Namespace``.
            run: Whether subcommands should be added to run a :class:`~pytorch_lightning.trainer.trainer.Trainer`
                method. If set to ``False``, the trainer and model classes will be instantiated only.
            auto_registry: Whether to automatically fill up the registries with all defined subclasses.
        """
        self.save_config_callback = save_config_callback
        self.save_config_kwargs = save_config_kwargs or {}
        self.trainer_class = trainer_class
        self.trainer_defaults = trainer_defaults or {}
        self.seed_everything_default = seed_everything_default
        self._raytune_callbacks = None

        self._handle_deprecated_params(kwargs)

        self.model_class = model_class
        # used to differentiate between the original value and the processed value
        self._model_class = model_class or LightningModule
        self.subclass_mode_model = (model_class is None) or subclass_mode_model

        self.datamodule_class = datamodule_class
        # used to differentiate between the original value and the processed value
        self._datamodule_class = datamodule_class or LightningDataModule
        self.subclass_mode_data = (datamodule_class is None) or subclass_mode_data

        from pytorch_lightning.utilities.cli import _populate_registries

        _populate_registries(auto_registry)

        main_kwargs, subparser_kwargs = self._setup_parser_kwargs(
            parser_kwargs or {},  # type: ignore  # github.com/python/mypy/issues/6463
            {
                "description": description,
                "env_prefix": env_prefix,
                "default_env": env_parse,
            },
        )
        self.setup_parser(run, main_kwargs, subparser_kwargs)
        self.parse_arguments(self.parser, args)

        self.subcommand = self.config["subcommand"] if run else None

        self._set_seed()

        # self.before_instantiate_classes()
        # self.instantiate_classes()

        # if self.subcommand is not None:
        #     self._run_subcommand(self.subcommand)

    def instantiate_and_run(self, hparams: Dict, raytune_callbacks: List[Callback]):
        use_custom_logger = self._rec_setattr(
            self.config, "trainer.logger.init_args.save_dir", tune.get_trial_dir()
        )
        self._raytune_callbacks = raytune_callbacks
        self._update_config(hparams)

        self.before_instantiate_classes()
        self.instantiate_classes()

        if not use_custom_logger:
            self.trainer.logger._save_dir = tune.get_trial_dir()

        if self.subcommand is not None:
            self._run_subcommand(self.subcommand)

    def instantiate_trainer(self, **kwargs: Any) -> Trainer:
        """Instantiates the trainer.

        Args:
            kwargs: Any custom trainer arguments.
        """
        extra_callbacks = [
            self._get(self.config_init, c)
            for c in self._parser(self.subcommand).callback_keys
        ]
        extra_callbacks.extend(self._raytune_callbacks)
        trainer_config = {**self._get(self.config_init, "trainer"), **kwargs}
        return self._instantiate_trainer(trainer_config, extra_callbacks)

    def _update_config(self, hparams: Dict):
        config = (
            getattr(self.config, self.subcommand)
            if self.subcommand and hasattr(self.config, self.subcommand)
            else self.config
        )

        # make the trainer quiet
        config.trainer.enable_progress_bar = False
        config.trainer.enable_model_summary = False

        for key, value in hparams.items():
            if not self._rec_setattr(config, key, value):
                raise AttributeError(
                    f"Did not manage to set the following hyperparameter: `{key}`"
                )

    @staticmethod
    def _rec_setattr(namespace, rec_attrs, val):
        n = namespace
        list_attrs = rec_attrs.split(".")
        last_attr = list_attrs.pop()
        for attr in list_attrs:
            if not hasattr(n, attr):
                return False
            n = getattr(n, attr)
        setattr(n, last_attr, val)
        return True
