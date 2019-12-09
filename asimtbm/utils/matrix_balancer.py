import logging
import numpy as np
from ipfn import ipfn

from activitysim.core import tracing


logger = logging.getLogger(__name__)

class Balancer(object):

    def __init__(self,
                 df,
                 aggregates,
                 dimensions,
                 weight_col,
                 convergence_rate=1e-6,
                 max_iteration=50,
                 closure=1e-4):
        """Wrapper for the IPFN matrix balancer
        https://github.com/Dirguis/ipfn



        Parameters
        ----------
        df : pandas DataFrame to balance
        aggregates : list of pandas groupby objects
            target sums
        dimensions : list of lists
            columns to sum over and compare to aggregates.
            length must match aggregates.
        weight_col : str
            data column in df
        convergence_rate : minimum difference between iterations
            to qualify as a change in convergence ratio
        closure : difference between convergence rate of two consecutive
            iterations to stop algorithm


        Returns
        -------
        Balancer object
        """

        self.df = df
        self.aggregates = aggregates
        self.dimensions = dimensions
        self.weight_col = weight_col
        self.convergence_rate = convergence_rate
        self.max_iteration = max_iteration
        self.closure = closure

        self.ipfn = ipfn.ipfn(
            self.df,
            self.aggregates,
            self.dimensions,
            weight_col=self.weight_col,
            convergence_rate=self.convergence_rate,
            max_iteration=self.max_iteration,
            rate_tolerance=self.closure,
            verbose=2
        )

    def balance(self):
        """Run an iteration of the balancer

        Returns
        -------
        Balanced pandas DataFrame
        """

        balanced_df, converged, info_df = self.ipfn.iteration()

        if not converged == 1:
            logger.warning('matrix balance failed to converge. See trace output table.')
            filename = self.weight_col + '_balancing_info'
            tracing.write_csv(info_df, filename, transpose=False)
        else:
            conv = info_df.iloc[-1]['conv']
            logger.info('success! matrix dimension difference converged to %s in %s iterations.'
                        % (conv, info_df.shape[0]))

        return balanced_df
