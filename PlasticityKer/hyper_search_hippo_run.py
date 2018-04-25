import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datajoint as dj
import hyper_search_hippo_schema as sk_hyperpara


if __name__ == "__main__":

    sk_hyperpara.ModelSelection().populate(reserve_jobs=True)