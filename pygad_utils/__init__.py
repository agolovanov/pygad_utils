import numpy as _np


class LinearTransform:
    def __init__(self, gene_range, value_range) -> None:
        """Defines a linear transformation between parameter space and gene space: $v = a g + b$.

        Parameters
        ----------
        gene_range : tuple of length 2
            the lower and upper limits of the gene space
        value_range : tuple of length 2
            the lower and upper limits of the parameter space
        """
        if len(gene_range) != 2:
            raise ValueError(f'The gene_range parameter should be a pair of values, not {gene_range}')
        if len(value_range) != 2:
            raise ValueError(f'The value_range parameter should be a pair of values, not {value_range}')

        g1, g2 = gene_range
        v1, v2 = value_range

        self.slope = (v2 - v1) / (g2 - g1)
        self.intercept = (v1 * g2 - v2 * g1) / (g2 - g1)

    def to_gene(self, value):
        import pint

        result = (value - self.intercept) / self.slope
        if isinstance(result, pint.Quantity):
            result = result.m_as('')

        return result

    def to_value(self, gene):
        return self.slope * gene + self.intercept


class ExponentialTransform:
    def __init__(self, gene_range, value_range) -> None:
        r"""Defines an exponential transformation between parameter space and gene space: $v = c \exp(a g)$.

        Parameters
        ----------
        gene_range : tuple of length 2
            the lower and upper limits of the gene space
        value_range : tuple of length 2
            the lower and upper limits of the parameter space
        """
        if len(gene_range) != 2:
            raise ValueError(f'The gene_range parameter should be a pair of values, not {gene_range}')
        if len(value_range) != 2:
            raise ValueError(f'The value_range parameter should be a pair of values, not {value_range}')

        g1, g2 = gene_range
        v1, v2 = value_range

        self.exp_coeff = _np.log(v2 / v1) / (g2 - g1)
        self.coeff = v2 ** (g1 / (g2 - g1)) * v1 ** (g2 / (g2 - g1))

    def to_value(self, gene):
        return self.coeff * _np.exp(self.exp_coeff * gene)

    def to_gene(self, value):
        return _np.log(value / self.coeff) / self.exp_coeff


class ParameterSpace:
    def __init__(self, parameters: dict, gene_range=None) -> None:
        """Defines a parameter space

        Parameters
        ----------
        parameters : dict
            a dictionary of either transforms or parameter spaces with a type of transform.
            ```
            {
                "density": (1e18, 1e20, 'exponential'),
                "length": (-1, 1, 'linear'),
                "size": (-5, -3),
                "scale": LinearTransform()
            }
            ```
            If a tuple of 3 is given, the first two values are the range of the parameter space, and the third is the
            type of the tranform to genes.
            If a tuple of 2 is given, the linear transform will be assumed.
        gene_range : tuple, optional
            The lower and upper limits of the gene space.
            Used to construct the transform.
            Required only if parameter spaces are provided instead of tranforms, by default None.
        """
        self.parameters = {}

        for k, parameter in parameters.items():
            to_value_function = getattr(parameter, "to_value", None)
            to_gene_function = getattr(parameter, "to_gene", None)
            if to_value_function is not None and to_gene_function is not None:
                self.parameters[k] = parameter
            else:
                if len(parameter) == 2:
                    type = 'linear'
                    v_min, v_max = parameter
                elif len(parameter) == 3:
                    v_min, v_max, type = parameter
                else:
                    raise ValueError(f'Unsupported type of parameter transformation {parameter}')
                if gene_range is None:
                    raise ValueError('gene_range has to be defined if not Transform objects are given')

                if type == 'lin' or type == 'linear':
                    self.parameters[k] = LinearTransform(gene_range, (v_min, v_max))
                elif type == 'exp' or type == 'exponential':
                    self.parameters[k] = ExponentialTransform(gene_range, (v_min, v_max))
                else:
                    raise ValueError(f'Unknown transform type {type},'
                                     'only exponential (exp) and linear (lin) are supported')

    def __len__(self):
        return len(self.parameters)

    def to_parameters(self, genes) -> dict:
        """Transforms the gene array to the parameter dictionary

        Parameters
        ----------
        genes : iterable
            the array of genes

        Returns
        -------
        dict
            the dictionary of the parameters corresponding to this gene
        """
        if len(genes) != len(self.parameters):
            raise ValueError(f'The size {len(genes)} of the gene {genes} is not equal '
                             f'to the size {len(self.parameters)} of the parameter space')

        result = {}
        for gene, k in zip(genes, self.parameters):
            result[k] = self.parameters[k].to_value(gene)

        return result

    def to_genes(self, parameters) -> tuple:
        return tuple(self.parameters[k].to_gene(parameters[k]) for k in self.parameters)
