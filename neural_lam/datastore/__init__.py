# Local
from .base import BaseDatastore  # noqa
from .mdp import MDPDatastore  # noqa
from .npyfilesmeps import NpyFilesDatastoreMEPS  # noqa

DATASTORE_CLASSES = [
    MDPDatastore,
    NpyFilesDatastoreMEPS,
]

DATASTORES = {
    datastore.SHORT_NAME: datastore for datastore in DATASTORE_CLASSES
}


def init_datastore(datastore_kind, **kwargs):
    DatastoreClass = DATASTORES.get(datastore_kind)

    if DatastoreClass is None:
        raise NotImplementedError(
            f"Datastore kind {datastore_kind} is not implemented"
        )

    datastore = DatastoreClass(**kwargs)

    return datastore
