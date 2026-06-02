from dataclasses import dataclass, field

from minio import Minio

from app.infra.config.providers import infra_config
from app.shared.clients.minio_utils import get_minio_client


@dataclass
class MinioGateway:

    bucket_name: str = infra_config.minio.bucket_name

    image_dir: str = infra_config.minio.minio_img_dir

    client: Minio = field(default_factory=get_minio_client)

    def build_image_url(self, stem: str, object_name: str):
        protocol = "https" if infra_config.minio.minio_secure else "http"

        return (
            f"{protocol}://{infra_config.minio.endpoint}/{self.bucket_name}"
            f"/{self.image_dir}/{stem}/{object_name}"
        )

minio_gateway = MinioGateway()

if __name__ == '__main__':
    url = MinioGateway().build_image_url("test", "test.png")

    print(url)

    client = MinioGateway().client

    buckets = client.list_buckets()

    print(buckets)
