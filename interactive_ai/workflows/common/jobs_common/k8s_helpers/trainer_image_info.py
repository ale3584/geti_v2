# Copyright (C) 2022-2025 Intel Corporation
# LIMITED EDGE SOFTWARE DISTRIBUTION LICENSE
"""
Defines train image info
"""

import asyncio
import logging
import os
from dataclasses import dataclass

from dataclasses_json import dataclass_json
from iai_core.entities.model import TrainingFramework, TrainingFrameworkType
from kubernetes_asyncio import client, config
from kubernetes_asyncio.config import ConfigException

logger = logging.getLogger(__name__)


async def get_config_map(namespace: str, name: str = "impt-configuration") -> client.V1ConfigMap:
    """
    Gets config map
    :param namespace: config map namespace
    :param name: config map name
    :return: config map
    """
    try:
        try:
            await config.load_kube_config()
        except (FileNotFoundError, TypeError, ConfigException):
            config.load_incluster_config()

        core_api = client.CoreV1Api(client.ApiClient())
        return await core_api.read_namespaced_config_map(namespace=namespace, name=name)

    except Exception:
        logger.exception(f"Failed to get config map {name} from {namespace} namespace")
        raise


@dataclass_json
@dataclass
class TrainerImageInfo:
    """Trainer image information."""

    train_image_name: str
    render_gid: int = 0  # Should be non-zero value when training with Intel GPUs

    @classmethod
    def create(cls, training_framework: TrainingFramework) -> "TrainerImageInfo":
        """Create trainer image information from the given training framework.

        :param training_framework: Dataclass to choose the trainer image
        """

        namespace = os.getenv("IMPT_NAMESPACE", "impt")
        name = os.getenv("IMPT_CONFIGURATION", "impt-configuration")

        configmap = asyncio.run(get_config_map(namespace=namespace, name=name))

        render_gid = 0
        msg = "Cannot get `{0}` field from config map `{1}/{2}`"

        if training_framework.type == TrainingFrameworkType.OTX:
            image_key = "otx2_image"

            if render_gid_value := configmap.data.get("render_gid"):
                render_gid = int(render_gid_value)

        elif (
            training_framework.type == TrainingFrameworkType.THIRD_PARTY
            and training_framework.version.startswith("ultralytics")
        ):
            image_key = "ultralytics_image"

        else:
            raise ValueError(
                f"Training framework {training_framework.type} "
                f"version {training_framework.version} is not supported yet."
            )

        if (image_name := configmap.data.get(image_key)) is None:
            raise ValueError(msg.format(image_key, namespace, name))

        logger.info(
            f"Trainer image has been selected {image_name}, "
            f"framework={training_framework.type}, "
            f"version={training_framework.version}."
        )

        return cls(
            train_image_name=image_name,
            render_gid=render_gid,
        )

    def to_image_full_name(self) -> str:
        """Get image full name.

        For example, it would be "dev-registry.toolbox.com/impp/ote:2.2.0".
        """
        return self.train_image_name
