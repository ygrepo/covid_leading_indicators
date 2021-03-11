import os
from pathlib import Path

import pandas as pd
import pyro
import pyro.contrib.gp as gp
import pyro.distributions as dist
import torch
from matplotlib import pyplot as plt


def plot(X, y, plot_observed_data=False, plot_predictions=False, n_prior_samples=0,
         model=None, kernel=None, n_test=500):
    plt.figure(figsize=(12, 6))
    if plot_observed_data:
        plt.plot(X.numpy(), y.numpy(), 'k-')
    if plot_predictions:
        Xtest = torch.linspace(X[-1], X[-1] + 10, n_test).double()  # test inputs
        # Xtest = torch.linspace(-0.5, 5.5, n_test).double()  # test inputs
        # compute predictive mean and variance
        with torch.no_grad():
            if type(model) == gp.models.VariationalSparseGP:
                mean, cov = model(Xtest, full_cov=True)
            else:
                mean, cov = model(Xtest, full_cov=True, noiseless=False)
        sd = cov.diag().sqrt()  # standard deviation at each input point x
        plt.plot(Xtest.numpy(), mean.numpy(), 'r', lw=2)  # plot the mean
        plt.fill_between(Xtest.numpy(),  # plot the two-sigma uncertainty about the mean
                         (mean - 1.0 * sd).numpy(),
                         (mean + 1.0 * sd).numpy(),
                         color='C0', alpha=0.3)
    if n_prior_samples > 0:  # plot samples from the GP prior
        Xtest = torch.linspace(X[0], X[-1] + 10, n_test).double()  # test inputs
        # Xtest = torch.linspace(-0.5, 5.5, n_test)  # test inputs
        noise = (model.noise if type(model) != gp.models.VariationalSparseGP
                 else model.likelihood.variance)
        cov = kernel.forward(Xtest) + noise.expand(n_test).diag().double()

        samples = dist.MultivariateNormal(torch.zeros(n_test).double(), covariance_matrix=cov) \
            .sample(sample_shape=(n_prior_samples,))
        plt.plot(Xtest.numpy(), samples.numpy().T, lw=2, alpha=0.4)

    #plt.xlim(0, 2)
    plt.show()


def main():
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

    pyro.enable_validation(__debug__)
    torch.manual_seed(0)
    report_frequency = 10

    path = "./data/projections/2020-05-23/US_NY.csv"
    ny_rt = pd.read_csv(path)
    r_values_mean = ny_rt["r_values_mean"]
    r_values_mean.dropna(inplace=True)
    X = torch.arange(0, r_values_mean.size, 1).double()
    y = torch.from_numpy(r_values_mean.values)

    kernel = gp.kernels.RBF(input_dim=1, variance=torch.tensor(5.), lengthscale=torch.tensor(10.))
    gpr = gp.models.GPRegression(X, y, kernel, noise=torch.tensor(1.))

    gpr.kernel.set_prior("lengthscale", dist.LogNormal(0.0, 1.0))
    gpr.kernel.set_prior("variance", dist.LogNormal(0.0, 1.0))
    # we reset the param store so that the previous inference doesn't interfere with this one
    pyro.clear_param_store()

    optimizer = torch.optim.Adam(gpr.parameters(), lr=0.005)
    loss_fn = pyro.infer.Trace_ELBO().differentiable_loss
    losses = []
    num_steps = 2500
    models_path = Path("models")

    def load_model(file_path):
        model_path = file_path / "saved_params.save"
        if model_path.exists():
            pyro.get_param_store().load(model_path)
            return True
        return False

    model_loaded = load_model(models_path)

    if not model_loaded:
        for step in range(num_steps):
            optimizer.zero_grad()
            loss = loss_fn(gpr.model, gpr.guide)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            if step % report_frequency == 0 or step == num_steps - 1:
                print("[step %03d]  loss: %.3f" % (step, loss.item()))

        plt.plot(losses)
        plt.show()
        plt.close()
        model_filename_path = models_path / "saved_params.save"
        pyro.get_param_store().save(model_filename_path)

    plot(X=X, y=y, model=gpr, plot_observed_data=True, plot_predictions=True, n_test=30, kernel=kernel,
         n_prior_samples=0)


if __name__ == "__main__":
    main()
