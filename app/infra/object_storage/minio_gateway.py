from dataclasses import dataclass

from app.infra.config.providers import InfraConfig
from app.shared.clients.minio_utils import get_minio_client


@dataclass
class MinioGateway:

    bucket_name: str = InfraConfig.minio.bucket_name

    image_dir: str = InfraConfig.minio.minio_img_dir

    client = get_minio_client()

    def build_image_url(self, stem: str, object_name: str):
        protocol = "https" if InfraConfig.minio.minio_secure else "http"

        return (
            f"{protocol}://{InfraConfig.minio.endpoint}/{self.bucket_name}"
            f"{self.image_dir}/{stem}/{object_name}"
        )
