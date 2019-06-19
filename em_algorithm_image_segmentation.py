# Import all needed modules
import numpy as np
from matplotlib import pyplot as plt
import copy

# Set a random seed for numpy
np.random.seed(1993)


# Method that calculates and returns the px (mixture distribution) and the nominator of the gammas using
# the pis (apriori probabilities), the means and the sigma squared values for all x and k
def get_px(x, pi, mi, sigma):
    # Get the N dimension from x
    N = x.shape[0]
    # Get the K dimension from pi
    K = pi.shape[0]
    # Initialize a NxK matrix of zeroes
    gam = np.zeros((N, K))
    # For k from 0 to K-1
    for k in range(K):
        # Compute the product that is in p(x) in the projectb.pdf as it is used for the gammas
        # Adding 1e-100 to denominators so that there isn't a division by 0 creating nans
        tmp1 = 1 / np.sqrt(2 * np.pi * sigma[k] + 1e-100)
        tmp2 = np.exp(-np.power((x - mi[k]), 2) / (2 * sigma[k] + 1e-100))
        gam[:, k] = pi[k] * np.prod(tmp1 * tmp2, axis=1)
    # Calculate the px (mixture distribution) by summing the gam matrix by K (axis 1)
    px = np.sum(gam, axis=1)
    # Return the px (mixture distribution) and the gam variable
    return px, gam


# Method that calculates and returns the px (mixture distribution) and the nominator of the gammas using
# the pis (apriori probabilities), the means and the sigma squared values for all x and k
# using logsumexp trick
def get_px_logsumexp_trick(x, pi, mi, sigma):
    # Get the N dimension from x
    N = x.shape[0]
    # Get the K dimension from pi
    K = pi.shape[0]
    # Initialize a NxK matrix of zeroes
    gam = np.zeros((N, K))
    # For k from 0 to K-1
    for k in range(K):
        # Compute the product that is in p(x) in the projectb.pdf as it is used for the gammas
        # Adding 1e-100 to denominators so that there isn't a division by 0 creating nans
        # using logsumexp_trick
        tmp1 = np.log(1 / np.sqrt(2 * np.pi * sigma[k] + 1e-100)) - np.power(x - mi[k], 2) / (2 * sigma[k] + 1e-100)
        gam[:, k] = np.log(pi[k]) + np.sum(tmp1, axis=1)
    # Get max for logsumexp_trick
    m = gam.max(axis=1)
    # Subtract max from gam
    for k in range(K):
        gam[:, k] = gam[:, k] - m
    # Pass gam from the exp function
    gam = np.exp(gam)
    # Calculate the px (mixture distribution) by summing the gam matrix by K (axis 1)
    px = np.sum(gam, axis=1)
    # Return the px (mixture distribution), gam variable and the maximum
    return px, gam, m


# Method that calculates and returns the gammas (aposteriori probabilities) using the nominator of the gammas
# and the p(x) mixture distribution
def get_gammas(gam, px):
    # Get the K dimension from pi
    K = gam.shape[1]
    for k in range(K):
        # Divide the gam matrix with the px for every k to get the gammas
        gam[:, k] = gam[:, k] / px
    # Returns the gammas and the px
    return gam


# Method that calculates and returns the new mean values using the new gammas for all k and d
def new_mkd(gammas, x):
    # Get the K dimension from gammas
    K = gammas.shape[1]
    # Get the D dimension from x
    D = x.shape[1]
    # Initialize a KxD matrix of zeroes
    mi = np.zeros((K, D))
    # Calculate the sum of the gammas by N (axis 0)
    sum_g = np.sum(gammas, axis=0)
    # For k from 0 to K-1
    for k in range(K):
        # Calculate the new means using the function as in lec7
        mi[k, :] = np.sum(gammas[:, k].reshape(-1, 1) * x, axis=0) / sum_g[k]
    # Return the new means
    return mi


# Method that calculates and returns the new sigma squared values using the new gammas and the new means for all k
def new_sk_d(gammas, x, mi):
    # Get the K dimension from gammas
    K = gammas.shape[1]
    # Get the D dimension from x
    D = x.shape[1]
    # Initialize a Kx1 matrix of zeroes
    sigma = np.zeros((K, 1))
    # Calculate the sum of the gammas by N (axis 0)
    sum_g = np.sum(gammas, axis=0)
    # For k from 0 to K-1
    for k in range(K):
        # Calculate the new sigma squared values using the function as in projectb.pdf
        sigma[k] = np.sum(gammas[:, k] * np.sum(np.power(x - mi[k], 2), axis=1)) / (D * sum_g[k])
    # Return the new sigma squared values
    return sigma


# Method that calculates the new pis (apriori probabilities) using the new gammas for all k
def new_pk(gammas):
    # Get the N dimenstion from gammas
    N = gammas.shape[0]
    # Get the K dimension from gammas
    K = gammas.shape[1]
    # Initialize a Kx1 matrix
    pi = np.zeros((K, 1))
    # For k from 0 to K-1
    for k in range(K):
        # Calculate the new pis (apriori probabilities) using the function as in lec7
        pi[k] = np.sum(gammas[:, k]) / N
    return pi


# Method that calculates and returns the log likelihood loss
def calc_loss(px):
    return np.sum(np.log(px))


# Method that calculates and returns the log likelihood loss
# using logsumexp trick
def calc_loss_logsumexp_trick(px, m):
    return np.sum(m + np.log(px))


# Method that calculates and returns the reconstruction error using the real pixel values and the
# ones that are replaced by the means
def calc_error(x_true, x_r):
    return np.sum((np.linalg.norm(x_true - x_r, axis=1) ** 2)) / x_true.shape[0]


# Main method that performs the EM Algorithm for the image segmentation using the arguments
def image_segmentation(img_filename='./im.jpg', K=2, plot_interval=-1, steps=300, thres=1e-2, show_figs=True,
                       save_figs=True, logsumexp_trick=True):
    # Load and read the image pixels using pyplot
    img = plt.imread(img_filename)
    # Reshape and normalize the image pixels to 0.0-1.0 float values
    x = np.asarray(img).reshape(-1, img.shape[2]) / 255
    # Create a deepcopy of the image pixels
    x_true = copy.deepcopy(x)
    # Find the N dimension from x
    N = len(x)
    # The D dimension is 3 - RGB
    D = 3
    # Initialize max for logsumexp trick
    m = None

    # Print info to the console
    print('-' * 30)
    print("Starting Image Segmentation with EM Algorithm")
    print("Image Shape:", img.shape)
    print("N:", N)
    print("K:", K)
    print("Steps:", steps)
    print("Threshold of convergence:", thres)
    print()

    # Initialize each pi value to 1/K
    pi = np.asarray([1 / K] * K)
    # Initialize mi, sigma squared to some random values from the uniform distribution
    mi = np.asarray([np.random.uniform(0.1, 0.9, D) for i in range(K)])
    sigma = np.random.uniform(0.2, 0.6, K)

    # Initialize a list for the error values
    errors = []
    # Initialize a list for the log likelihood values
    logL = []
    # Initialize a list for the log likelihood diff values
    logLdiffs = []
    # Initialize a variable for the new iamge pixel values to None
    new_x = None
    # Append the minus infinity to the log likelihood list
    logL.append(float('-inf'))
    # Get the p(x) mixture distribution and the nominator of the gammas
    if logsumexp_trick:
        px, gammas, m = get_px_logsumexp_trick(x, pi, mi, sigma)
    else:
        px, gammas = get_px(x, pi, mi, sigma)
    # For the steps specified by the arguments
    for s in range(steps):

        # Print the STEP to the console
        print('=' * 30, 'STEP %s' % s, '=' * 30)

        """ Expectation step """

        # Print to the console that the algorithm is in the expectation step
        print('-' * 27, "Expectation ", '-' * 27)
        # Calculate the new gamma values and the px mixture distribution
        gammas = get_gammas(gammas, px)
        # Print the new gamma values
        # print('gamma values: ', gammas)
        print('\nCalculated new gamma values\n')

        """ Maximization step """

        # Print to the console that the algorithm is in the expectation step
        print('-' * 27, "Maximization", '-' * 27)
        # Assign new mean values
        mi = new_mkd(gammas, x)
        # Assign new sigma squared values
        sigma = new_sk_d(gammas, x, mi)
        # Assign new pi values
        pi = new_pk(gammas)

        # Print the new vals of mi, sigma squared and pi values
        # print('mi values: %s\nsigma squared values: %s\npi values: %s' % (
        #     str([e for e in mi]), str([e for e in sigma]), str([e for e in pi])))
        print('\nCalculated new mi, sigma squared and pi values\n')

        """ LogLikelihood and Reconstruction Error """
        # Print to the console that the algorithm is in the expectation step
        print('-' * 27, "LogL / Error", '-' * 27)

        # Calculate the px (mixture distribution) and the nominator of the gammas
        if logsumexp_trick:
            px, gammas, m = get_px_logsumexp_trick(x, pi, mi, sigma)
        else:
            px, gammas = get_px(x, pi, mi, sigma)
        # Calculate the log likelihood loss and append it to the list
        if logsumexp_trick:
            logL.append(calc_loss_logsumexp_trick(px, m))
        else:
            logL.append(calc_loss(px))
        # Calculate the log likelihood loss difference from the previous step
        logL_diff = logL[-1] - logL[-2]
        # Append the log likelihood difference to the list
        logLdiffs.append(logL_diff)
        # Print the log likelihood and the difference from the previous step to the console
        print("\nLog Likelihood: %f\nDiff: %f" % (logL[-1], logL_diff))

        # Get the new image pixels using the correct mean value in regard to the argmax index k for each x
        # from the gammas
        new_x = [[mi[i][0], mi[i][1], mi[i][2]] for i in
                 np.argmax(np.asarray(gammas), axis=1)]

        # Calculate the reconstruction error and append it to the list
        errors.append(calc_error(x_true, new_x))

        # Print the error to the console
        print("Reconstruction Error: %f\n" % errors[-1])

        # Show image every plot_interval iterations if plot_interval is not -1
        if plot_interval != -1 and s % plot_interval == 0:
            # Reshape the new image
            new_img = np.asarray(new_x).reshape(img.shape[0], img.shape[1], img.shape[2])
            # Disable the axis on the plot
            plt.axis('off')
            # Load the image on the plot
            plt.imshow(new_img)
            # Show the image
            plt.show()

        # Check that the log loss is increasing and check for threshold convergence
        if logL_diff < 0:
            print('ERROR: Log Likelihood is not increasing!!')
            exit(1)
        if logL_diff < thres:
            # Print a message to the console
            print('@' * 17, 'The Log Likelihood has converged', '@' * 17)
            # Break the loop
            break

    # Delete the first logL value of -inf
    logL.pop(0)
    # Create a log with the LogL, Diff and Reconstruction Error lists
    with open(img_filename[:-4] + '_K_' + str(K) + "_log.txt", 'w') as logfile:
        # Write to logfile the contents of logL list
        logfile.write('Log Likelihood List\n')
        logfile.write(str(logL))
        logfile.write('\n')
        # Write to logfile the contents of logLdiffs list
        logfile.write('Log Likelihood Diff List\n')
        logfile.write(str(logLdiffs))
        logfile.write('\n')
        # Write to logfile the contents of errors list
        logfile.write('Reconstruction Error List\n')
        logfile.write(str(errors))
        logfile.write('\n')

    # Reshape the new image
    new_img = np.asarray(new_x).reshape(img.shape[0], img.shape[1], img.shape[2])

    if show_figs:
        """ Plot end image """
        # Create a new figure
        plt.figure()
        # Disable the axis on the plot
        plt.axis('off')
        # Load the image on the plot
        plt.imshow(new_img)
        # Show the image
        plt.show()
        """ Plot log likelihood """
        # Create a new figure
        plt.figure()
        # Plot log likelihood
        plt.plot(logL, label='Log Likelihood')
        # Plot log likelihood diff
        plt.plot(logLdiffs, label='Diff')
        # Change x,y labels
        plt.xlabel('Iterations')
        plt.ylabel('Log Likelihood')
        # Enable legend on the plot
        plt.legend()
        # Show the image
        plt.show()
        """ Plot reconstruction error """
        # Create a new figure
        plt.figure()
        # Plot reconstruction error
        plt.plot(errors, label='Reconstruction Error')
        # Change x,y labels
        plt.xlabel('Iterations')
        plt.ylabel('Reconstruction Error')
        # Show the image
        plt.show()
    if save_figs:
        """ Save end image """
        # Create a new figure
        plt.figure()
        # Load the image on the plot
        fig = plt.imshow(new_img)
        # Disable the axis on the plot
        plt.axis('off')
        # Set the figure x and y axis invisible
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)
        # Save the image
        plt.savefig(img_filename[:-4] + '_K_' + str(K) + '.png', bbox_inches='tight', pad_inches=0)
        """ Save log likelihood plot """
        # Create a new figure
        plt.figure()
        # Plot log likelihood
        plt.plot(logL, label='Log Likelihood')
        # Plot log likelihood diff
        plt.plot(logLdiffs, label='Diff')
        # Change x,y labels
        plt.xlabel('Iterations')
        plt.ylabel('Log Likelihood')
        # Enable legend on the plot
        plt.legend()
        # Save the image
        plt.savefig(img_filename[:-4] + '_K_' + str(K) + '_logLplot.png', bbox_inches='tight', pad_inches=0)
        """ Save reconstruction error plot"""
        # Create a new figure
        plt.figure()
        # Plot reconstruction error
        plt.plot(errors, label='Reconstruction Error')
        # Change x,y labels
        plt.xlabel('Iterations')
        plt.ylabel('Reconstruction Error')
        # Show the image
        plt.savefig(img_filename[:-4] + '_K_' + str(K) + '_errorPlot.png', bbox_inches='tight', pad_inches=0)


""" Main code """

# Run the image segmentation algorithm for different values of K
for k in [1, 2, 4, 8, 16, 32, 64]:
    image_segmentation(K=k, thres=0.1, logsumexp_trick=True)
