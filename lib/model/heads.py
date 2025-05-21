import torch
import torch.nn as nn


class MegaAgeLayer(nn.Module):
    """
        Implements the prediction head from "Quantifying Facial Age by Posterior of Age Comparisons".

        The features are first processed by a linear layer and (nr_classes-1) logits corresponding to extended binary classification (EBC) task are predicted.
        Sigmoid of the binary logits is taken and the output is processed by a linear layer, outputing (nr_classes) logits corresponding to the class posterior.

        The binary probabilities and the output logits are concatenated and returned.
        This is done, because the loss is computed over both: 1) the EBC, 2) the output posterior.
    """

    def __init__(self, features_in, n_classes):
        """
        Args:
            features_in (int): Number of features extracted for each image.
            n_classes (int): Number of classes, i.e., label space cardinality.
        """
        super().__init__()
        self.fc1 = nn.Linear(features_in, n_classes-1, bias=True)
        self.fc2 = nn.Linear(n_classes-1, n_classes, bias=True)

    def forward(self, x):
        """
        Forward pass of the layer.

        The features are first processed by a linear layer and (nr_classes-1) logits corresponding to extended binary classification (EBC) task are predicted.
        Sigmoid of the binary logits is taken and the output is processed by a linear layer, outputing (nr_classes) logits corresponding to the class posterior.

        The binary probabilities and the output logits are concatenated and returned.
        This is done, because the loss is computed over both: 1) the EBC, 2) the output posterior.
        """
        binary_probas = torch.sigmoid(self.fc1(x))
        logits = self.fc2(binary_probas)
        return torch.cat([logits, binary_probas], dim=-1)


class CoralLayer(nn.Module):
    """
        Implements the prediction head from Rank consistent ordinal regression for neural networks with application to age estimation.
    """

    def __init__(self, features_in, n_classes, preinit_bias=True):
        """
        Args:
            features_in (int): Number of features extracted for each image.
            n_classes (int): Number of classes, i.e., label space cardinality.
            preinit_bias (bool, optional): If True, the biases are initialized to an ordered sequence. Defaults to True. 
        """
        super().__init__()
        self.coral_weights = nn.Linear(features_in, 1, bias=False)
        if preinit_bias:
            self.coral_bias = nn.Parameter(torch.arange(n_classes-1, 0, -1).float() / (n_classes-1))
        else:
            self.coral_bias = nn.Parameter(torch.zeros(n_classes-1).float())

    def forward(self, x):
        """
        Forward pass of the CORAL model. The weight vector of the logits is shared, but different biases are used.
        """
        return self.coral_weights(x) + self.coral_bias
    

def get_head_posterior(logits, head_type: str):
        """
        Method which computes the posterior probabiility of each class for a particular prediction head.
        All implemented methods need to define a posterior of the age, given the image.
        For a standard classification head, the posterior is computed by applying softmax to the logits.
        The same is true for some other methods, e.g., Mean-Variance loss or DLDL.

        Some methods do not predict a posterior, but instead predict the age itself.
        For such methods, e.g., regression, we construct the posterior probability as equal to 1 at the predicted age and 0 elsewhere.        

        We enforce that each method computes a posterior so that        For other methods, e.g., CORAL, the loss is implemented in the :py:mod:`lib.loss` module. it enables unified further evaluations.

        Args:
            logits (torch.tensor): Output of the prediction head.
            tag (str): Unique identifier of the prediction head. 
                Based on the tag, the prediction head type is determined.
                The method of computing the posterior is then decided based on the method/head type.

        Raises:
            NotImplementedError: If type of the specified head is not implemented.
        """

        if head_type in [ 'classification', 'dldl', 'dldl_v2', 'unimodal_concentrated', 'soft_labels', 'mean_variance' ]:
            return torch.softmax(logits, 1)
        
        elif head_type in [ 'regression' ]:
            # encode the prediction as one-hot posterior
            logits = logits.flatten()
            predicted_labels = torch.minimum(
                torch.maximum(torch.round(logits), torch.zeros_like(logits)), 
                torch.ones_like(logits)*(self.n_classes-1)
            ).long()
            posterior = F.one_hot(predicted_labels, num_classes=self.n_classes).type(logits.dtype)
            return posterior
        
        elif head_type in [ 'megaage' ]:
            return torch.softmax(logits[:, :self.n_classes], 1)
        
        elif head_type in [ 'orcnn', 'extended_binary_classification' ]:
            # encode the prediction as one-hot posterior
            binary_probas = torch.sigmoid(logits)
            predicted_labels = torch.sum(binary_probas > 0.5, dim=1)
            posterior = F.one_hot(predicted_labels, num_classes=self.n_classes).type(logits.dtype)
            return posterior
        
        elif head_type in [ 'coral' ]:
            binary_probas = torch.sigmoid(logits)
            # computes (1, p[0], p[1], ..., p[K]) - (p[0], p[1], ..., p[K], 0)
            # i.e., computes (1 - p[0], p[0]-p[1], ..., p[K] - 0)
            # for each sample in the mini batch
            A = torch.cat([
                torch.ones(binary_probas.shape[0], 1).type(binary_probas.dtype).to(binary_probas.device), 
                binary_probas
            ], dim=1)
            B = torch.cat([
                binary_probas, 
                torch.zeros(binary_probas.shape[0], 1).type(binary_probas.dtype).to(binary_probas.device)
            ], dim=1)
            # posterior, but not yet normalized
            pseudo_posterior = A - B
            # subtract minimal value
            min_vals, ixs = torch.min(pseudo_posterior, dim=1, keepdim=True)
            pseudo_posterior = pseudo_posterior - torch.broadcast_to(min_vals, pseudo_posterior.shape)
            # normalize
            return pseudo_posterior / torch.broadcast_to(torch.sum(pseudo_posterior, dim=1, keepdim=True), pseudo_posterior.shape)
        
        else:
            raise NotImplementedError()