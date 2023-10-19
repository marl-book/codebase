from collections import defaultdict

import click
import hiplot as hip
import json
import numpy as np
import pandas as pd


class NumpyEncoder(json.JSONEncoder):
    """ Custom encoder for numpy data types """
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):

            return int(obj)

        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)

        elif isinstance(obj, (np.complex_, np.complex64, np.complex128)):
            return {'real': obj.real, 'imag': obj.imag}

        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()

        elif isinstance(obj, (np.bool_)):
            return bool(obj)

        elif isinstance(obj, (np.void)): 
            return None

        return json.JSONEncoder.default(self, obj)

def experiment_fetcher(uri):

    PREFIX = "exp://"

    if not uri.startswith(PREFIX):
        # Let other fetchers handle this one
        raise hip.ExperimentFetcherDoesntApply()
    uri = uri[len(PREFIX):]  # Remove the prefix

    exported_file = uri.split("/")[0]

    df = pd.read_hdf(exported_file, "df")
    configs = pd.read_hdf(exported_file, "configs")

    df = (
        df.groupby(axis=1, level=[0, 1, 2]).mean().max()
    )

    data = defaultdict(lambda: defaultdict(list))

    for env, df in df.groupby(level=0):
        df = df.xs(env)
        for alg, df in df.groupby(level=0):
            df = df.xs(alg)

            for index, perf in df.iteritems():
                data[env][alg].append({**configs.loc[index].to_dict(), "performance": perf, "uid": index})

    env, alg = uri.split("/")[1], uri.split("/")[2]

    data = json.loads(json.dumps(data[env][alg], cls=NumpyEncoder))
    exp = hip.Experiment.from_iterable(data)

    return exp
    

if __name__ == "__main__":
    click.echo("Run with \"hiplot fastmarl.utils.postprocessing.hiplot_fetcher.experiment_fetcher\"")
    click.echo("And enter \"exp://filename.hd5/envname/alg\" in the textbox")

