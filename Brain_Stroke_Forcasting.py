import pandas
import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from enum import Enum
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import statsmodels.api as sm
import seaborn as sns  # Used for correlation Plot and Count Plot
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import auc
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import RocCurveDisplay

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde


class Algorithm(Enum):
    SVM_RBF = 1
    DECISION_TREE = 2
    K_NN = 3
    NAIVE_BAYES = 4
    LINEAR_SVM = 5
    LOGISTIC_REGRESSION = 6


class ResampleType(Enum):
    DISABLE = 0
    UNDER   = 1
    OVER    = 2


def get_classifier(input_test_algorithm):
    if input_test_algorithm == Algorithm.NAIVE_BAYES:
        return GaussianNB()
    elif input_test_algorithm == Algorithm.LINEAR_SVM:
        return SVC(C=1, kernel='linear', gamma = 0.001)
    elif input_test_algorithm == Algorithm.LOGISTIC_REGRESSION:
        return LogisticRegression(solver='liblinear', random_state=123)
    elif input_test_algorithm == Algorithm.SVM_RBF:
        return SVC(C=1, kernel='rbf', gamma=0.1)
    elif input_test_algorithm == Algorithm.DECISION_TREE:
        return DecisionTreeClassifier(random_state=123)
    elif input_test_algorithm == Algorithm.K_NN:
        return KNeighborsClassifier(n_neighbors=11, weights="distance")
    else:
        return GaussianNB()

def get_classifier_name(input_test_algorithm):
    if input_test_algorithm == Algorithm.NAIVE_BAYES:
        return "GaussianNB"
    elif input_test_algorithm == Algorithm.LINEAR_SVM:
        return "Linear SVC"
    elif input_test_algorithm == Algorithm.LOGISTIC_REGRESSION:
        return "Logistic Regression"
    elif input_test_algorithm == Algorithm.SVM_RBF:
        return "SVC RBF"
    elif input_test_algorithm == Algorithm.DECISION_TREE:
        return "Decision Tree"
    elif input_test_algorithm == Algorithm.K_NN:
        return "K-NN"
    else:
        return "GaussianNB"

def get_classifier_color(input_test_algorithm):
    if input_test_algorithm == Algorithm.NAIVE_BAYES:
        return "b"
    elif input_test_algorithm == Algorithm.LINEAR_SVM:
        return "c"
    elif input_test_algorithm == Algorithm.LOGISTIC_REGRESSION:
        return "r"
    elif input_test_algorithm == Algorithm.SVM_RBF:
        return "g"
    elif input_test_algorithm == Algorithm.DECISION_TREE:
        return "y"
    elif input_test_algorithm == Algorithm.K_NN:
        return "k"
    else:
        return "b"
# Clean Stroke data
def clean_stroke_data(input_data):
    input_data = input_data.drop(columns=["id"])
    input_data["gender"].replace(["Male", "Female", "Other"], [1, 0, 2], inplace=True)
    input_data["ever_married"].replace(["Yes", "No"], [1, 0], inplace=True)
    input_data["work_type"].replace(["children", "Never_worked", "Govt_job", "Private", "Self-employed"],[0, 1, 2, 3, 4], inplace=True)
    input_data["Residence_type"].replace(["Urban", "Rural"], [0, 1], inplace=True)
    input_data["smoking_status"].replace(["formerly smoked", "never smoked", "smokes", "Unknown"], [0, 1, 2, 3],inplace=True)
    input_data = input_data.dropna(inplace=False)
    input_data = input_data.reset_index(drop=True)

    return input_data


def find_optimal_k_for_KNN(input_x_data, input_y_data):
    # empty variable for storing the KNN metrics
    scores = []
    # We try different values of k for the KNN (from k=1 up to k=20)
    lrange = list(range(1, 20))
    # loop the KNN process
    for k in lrange:
        # input the k value and 'distance' measure
        knn = KNeighborsClassifier(n_neighbors=k, weights='distance', algorithm='auto')

        scores_from_knn, best_col_index = score_single_dimension_test(knn, input_x_data, input_y_data)

        # append the performance metric (accuracy)
        print(f"KNN score: {scores_from_knn}")
        scores.append(scores_from_knn)
    optimal_k = lrange[scores.index(max(scores))]

    print("The optimal number of neighbors is %d" % optimal_k)
    print("The optimal score is %.2f" % max(scores))
    plt.figure(2, figsize=(15, 5))

    # plot the results
    plt.plot(lrange, scores, ls='dashed')
    plt.xlabel('Value of k for KNN')
    plt.ylabel('Accuracy Score')
    plt.title('Accuracy Scores for Values of k of k-Nearest-Neighbors')
    plt.xticks(lrange)
    plt.savefig("images/BestKNN.jpg")
    plt.show()

    return optimal_k


def plot_svc_decision_function(model, ax=None, plot_support=True):
    if ax is None:
        ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # create grid to evaluate model
    x = np.linspace(xlim[0], xlim[1], 30)
    y = np.linspace(ylim[0], ylim[1], 30)
    Y, X = np.meshgrid(y, x)
    xy = np.vstack([X.ravel(), Y.ravel()]).T
    P = model.decision_function(xy).reshape(X.shape)

    # plot decision boundary and margins
    ax.contour(X, Y, P, colors='k',
                levels=[-1, 0, 1], alpha=0.5,
                linestyles=['--', '-', '--'])

    # plot support vectors
    if plot_support:
        ax.scatter(model.support_vectors_[:, 0],
                    model.support_vectors_[:, 1],
                    s=300, linewidth=1, facecolors='none');
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)


def generate_mean_ROC_curve_for_classifiers(classifier_list,
                                            df_all_data_input,
                                            save_fig_name = None):
    single_data_column_label = "age"
    number_of_folds = 9


    df_all_x_data = df_all_data_input.drop(["gender","Residence_type"],axis=1)
    df_all_y_data = df_all_data_input.iloc[:, -1]

    # Set Seeds for Resampling
    under_sampler = RandomUnderSampler(random_state=42)

    # Resample All Data
    x_all_data_under_sample, y_all_data_under_sample = under_sampler.fit_resample(df_all_x_data, df_all_y_data)


    fig_size_tuple = (5, 5)
    target_names = x_all_data_under_sample.columns

    cross_validation = StratifiedKFold(n_splits=number_of_folds)

    # x_data_roc = df_all_x_data.to_numpy()  # x_all_data_under_sample["age"].to_numpy()
    # x_data_roc = x_data_roc.reshape(-1, 1)
    x_data_roc = x_all_data_under_sample.to_numpy()
    # y_data_roc = df_all_y_data.to_numpy()
    y_data_roc = y_all_data_under_sample.to_numpy()

    figure_all_mean, axis_all_mean = plt.subplots(figsize=fig_size_tuple)
    for classifier_type in classifier_list:
        classifier       = get_classifier(classifier_type)
        classifier_name  = get_classifier_name(classifier_type)
        classifier_color = get_classifier_color(classifier_type)
        figure, axis = plt.subplots(figsize=fig_size_tuple)

        true_positive_rate_list = []
        area_under_curve_list = []
        mean_fpr_list = np.linspace(0, 1, 100)

        for fold, (train_data_ind, test_data_ind) in enumerate(cross_validation.split(x_data_roc, y_data_roc)):
            x_train_classifier = x_data_roc[train_data_ind]
            x_test_classifier = x_data_roc[test_data_ind]

            y_train_classifier = y_data_roc[train_data_ind]
            y_test_classifier = y_data_roc[test_data_ind]

            classifier.fit(x_train_classifier, y_train_classifier)
            visualize = RocCurveDisplay.from_estimator(classifier,
                                                       x_test_classifier,
                                                       y_test_classifier,
                                                       name=f"Fold {fold}",
                                                       alpha=0.3,
                                                       lw=1,
                                                       ax=axis)
            interpolate_true_positive_rate = np.interp(mean_fpr_list, visualize.fpr, visualize.tpr)
            interpolate_true_positive_rate[0] = 0.0
            true_positive_rate_list.append(interpolate_true_positive_rate)
            area_under_curve_list.append(visualize.roc_auc)

        average_true_positive_rate = np.mean(true_positive_rate_list, axis=0)
        average_true_positive_rate[-1] = 1.0
        average_area_under_curve = auc(mean_fpr_list, average_true_positive_rate)
        axis_all_mean.plot(mean_fpr_list,
                  average_true_positive_rate,
                  color=classifier_color,
                  label=rf"{classifier_name} - Mean ROC: {round(average_area_under_curve,4)}",
                  lw=2,
                  alpha=0.8)
    axis_all_mean.plot([0, 1], [0, 1], "k--", label="AUC = 0.5")
    axis_all_mean.set(
        xlim=[-0.05, 1.05],
        ylim=[-0.05, 1.05],
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
        title=f"Mean ROC curve. {number_of_folds} Fold CV\n(Positive label Stroke')",
    )
    axis_all_mean.axis("square")
    axis_all_mean.legend(loc="lower right")
    plt.figure(figure_all_mean.number)
    if save_fig_name is not None:
        plt.savefig(save_fig_name)
    plt.show()


def plot_k_fold_roc_curve(test_algorithm, x_data_in, y_data_in, resample_type_in = None, save_name = None):
    number_of_folds = 5
    fig_size_tuple = (number_of_folds, number_of_folds)
    target_names = x_data_in.columns

    cross_validation = StratifiedKFold(n_splits=number_of_folds)
    true_positive_rate_list = []
    area_under_curve_list   = []
    mean_fpr_list = np.linspace(0, 1, 100)

    if resample_type_in == ResampleType.UNDER:
        # Set Seeds for Resampling
        under_sampler = RandomUnderSampler(random_state=42)

        x_data_in, y_data_in = under_sampler.fit_resample(x_data_in, y_data_in)

    x_data_roc = x_data_in.to_numpy()
    # x_data_roc = x_data_roc.reshape(-1,1)
    y_data_roc = y_data_in.to_numpy()

    classifier_str = get_classifier_name(test_algorithm)
    classifier = get_classifier(test_algorithm)
    figure, axis = plt.subplots(figsize=fig_size_tuple)
    for fold, (train_data_ind, test_data_ind) in enumerate(cross_validation.split(x_data_roc, y_data_roc)):
        x_train_classifier = x_data_roc[train_data_ind]
        x_test_classifier  = x_data_roc[test_data_ind]

        y_train_classifier = y_data_roc[train_data_ind]
        y_test_classifier  = y_data_roc[test_data_ind]

        classifier.fit(x_train_classifier, y_train_classifier)
        visualize = RocCurveDisplay.from_estimator(classifier,
                                                   x_test_classifier,
                                                   y_test_classifier,
                                                   name=f"Fold {fold}",
                                                   alpha=0.3,
                                                   lw=1,
                                                   ax=axis)
        interpolate_true_positive_rate = np.interp(mean_fpr_list, visualize.fpr, visualize.tpr)
        interpolate_true_positive_rate[0] = 0.0
        true_positive_rate_list.append(interpolate_true_positive_rate)
        area_under_curve_list.append(visualize.roc_auc)
    axis.plot([0,1], [0,1], "k--", label="AUC = 0.5")

    average_true_positive_rate = np.mean(true_positive_rate_list, axis=0)
    average_true_positive_rate[-1] = 1.0
    average_area_under_curve = auc(mean_fpr_list, average_true_positive_rate)
    std_area_under_curve = np.std(area_under_curve_list)
    axis.plot(mean_fpr_list,
              average_true_positive_rate,
              color="b",
              label=f"Mean ROC: {round(average_area_under_curve,2)}",
              lw=2,
              alpha=0.8)

    axis.set(
        xlim=[-0.05, 1.05],
        ylim=[-0.05, 1.05],
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
        title=f" {classifier_str} mean ROC curve \n(Positive label 'Stroke')",
    )
    axis.axis("square")
    axis.legend(loc="lower right")
    if save_name is not None:
        plt.savefig(save_name)
    plt.show()



# Plot Linear SVC
def plot_linear_svc(x_data, y_data, title = ""):
    #fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    fig = plt.figure()
    #fig.subplots_adjust(left=0.0625, right=0.95, wspace=0.1)
    # test two different 'c' values (10 and 0.1) and plot the results
    #for axi, C in zip(ax, [1.0]):
    model = SVC(kernel='linear', C=1.0).fit(x_data, y_data)
    #axi.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')
    s_map = y_data
    s_map[[i for i,v in enumerate(y_data) if v > 0.5]] = 20
    s_map[[i for i,v in enumerate(y_data) if v < 0.5]] = 80
    plt.scatter(x_data[:, 0], x_data[:, 1], c=y_data, s=s_map, cmap='bwr')
    plot_svc_decision_function(model)
    plt.scatter(model.support_vectors_[:, 0],
                model.support_vectors_[:, 1],
                s=300, lw=1, facecolors='none')
    plt.title(title, size=14)


# Creates a Correlation Plot
def plot_correlation_data(input_data, title, save_name = None):
    fig_length = len(input_data.columns)
    fig_height = len(input_data.columns)

    plt.figure(figsize=(fig_length, fig_height))
    ax = sns.heatmap(input_data.corr(), annot=True, cmap='summer', cbar=True)
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)
    plt.title(title)

    if save_name is not None:
        plt.savefig(save_name)

    plt.show()


# Plots count of dataset
def plot_value_count(input_data,
                     title="",
                     save_name=None):
    # Before over sample
    fig_length = 5
    fig_height = 5

    plt.figure(figsize=(fig_height, fig_length))
    plt.title(title)

    # count rows of each classes
    sns.countplot(x=input_data)

    plt.show()

    if save_name is not None:
        plt.savefig(save_name)


# def score_single_dimension_test(classifier, x_train, y_train, x_test, y_test, enable_scaling = False):
def score_single_dimension_test(classifier, x_data_in, y_data, enable_scaling=False):
    col_1_index = 0
    col_1_index_end = len(x_data_in.columns)
    current_score = 0
    best_score = 0
    best_score_col_1_index = 0
    while col_1_index < col_1_index_end:
        x_data = x_data_in.iloc[:, [col_1_index]]

        # Cross Validata
        current_score = cross_val_score(classifier,x_data, y_data).mean() * 100

        if current_score > best_score:
            best_score_col_1_index = col_1_index
            best_score = current_score
        #print(f"{x_data_in.columns[col_1_index]} Accuracy {round(current_score,2)}")
        col_1_index += 1

    print(f"Best results found using {x_data_in.columns[best_score_col_1_index]}")
    return best_score, best_score_col_1_index


# def score_two_dimension_test(classifier_in, x_train, y_train, x_test, y_test, enable_scaling = False):
def score_two_dimension_test(classifier_in, x_data_in, y_data_in, enable_scaling=False):
    # start algorithm
    col_1_data = 0
    col_2_data_end = len(x_data_in.columns)
    col_1_data_end = col_2_data_end - 1

    current_score = 0
    best_score = 0
    best_score_col_1_data = 0
    best_score_col_2_data = 1

    # print("In two dimension test")
    while col_1_data < col_1_data_end:
        col_2_data = col_1_data + 1
        while col_2_data < col_2_data_end:

            # print("Selecting Data")
            x_data = x_data_in.iloc[:, [col_1_data, col_2_data]]
            y_data = y_data_in

            # x_test_classifier = x_test.iloc[:, [col_1_data, col_2_data]]
            # y_test_classifier = y_test

            if enable_scaling:
                # print("Scaling data")
                scale_value = MinMaxScaler(feature_range=(-1, 1)).fit(x_data)
                x_data = scale_value.transform(x_data)

            # Call Classifier
            # print("Calling Classifier")
            current_score = cross_val_score(classifier_in, x_data, y_data, cv=9, scoring='accuracy').mean() * 100

            # print(f"Testing {df_all_data.columns[col_1_data]} and {df_all_data.columns[col_2_data]}: {round(current_score, 2)}")

            # Generic
            if current_score > best_score:
                best_score_col_1_data = col_1_data
                best_score_col_2_data = col_2_data
                best_score = current_score

            col_2_data += 1
        col_1_data += 1
    #print("The best default prediction accuracy is: {0:2.2f}{1:s}".format(best_score, "%"))
    print(f"Best results found using {x_data_in.columns[best_score_col_1_data]} and {x_data_in.columns[best_score_col_2_data]}")

    return best_score, [best_score_col_1_data, best_score_col_2_data]


def score_all_dimensions_test(classifier_in, x_data_in, y_data_in, enable_scaling=False, cross_validations = 9):
    # start algorithm
    x_data = x_data_in
    y_data = y_data_in

    if enable_scaling:
        # print("Scaling data")
        scale_value = MinMaxScaler(feature_range=(-1, 1)).fit(x_data)
        x_data = scale_value.transform(x_data)

    # Call Classifier, Cross Validate, and Score
    current_score = cross_val_score(classifier_in, x_data, y_data, cv=cross_validations, scoring='accuracy').mean() * 100

    return current_score, ""


def show_likelihood_of_stroke_if_ever_married(input_x_data, input_y_data):
    x = input_x_data["ever_married"]
    y = input_y_data

    married_stroke = 0
    not_married_stroke = 0

    married_no_stroke = 0
    not_married_no_stroke = 0

    for data in zip(x, y):
        if data == (1, 1):
            married_stroke += 1
        elif data == (1, 0):
            married_no_stroke += 1
        elif data == (0, 1):
            not_married_stroke += 1
        else:
            not_married_no_stroke += 1
    print(f"{married_stroke/(married_stroke + married_no_stroke)}")
    print(f"{married_no_stroke/(married_stroke + married_no_stroke)}")
    print(f"{not_married_stroke/(not_married_stroke + not_married_no_stroke)}")
    print(f"{not_married_no_stroke/(not_married_stroke + not_married_no_stroke)}")


def show_logistic_regression_table(x_data, y_data):
    reg_table_data = LogisticRegression(solver='liblinear', random_state=123).fit(x_data, y_data)
    #print(reg_table_data.coef_, reg_table_data.intercept_)
    print(sm.Logit(y_data, x_data).fit().summary())


def test_algorithm_with_data(test_algorithm,
                             df_all_data_input,
                             use_minimal_variables_in=False,
                             resample_type_in=ResampleType.UNDER):
    df_all_x_data = df_all_data_input.iloc[:,:-1]
    df_all_y_data = df_all_data_input.iloc[:,-1]

    # Set Seeds for Resampling
    over_sampler = RandomOverSampler(random_state=42)
    #under_sampler = RandomUnderSampler(random_state=42)
    under_sampler = RandomUnderSampler(random_state=42)

    # Resample All Data
    x_all_data_under_sample, y_all_data_under_sample = under_sampler.fit_resample(df_all_x_data, df_all_y_data)
    x_all_data_over_sample,  y_all_data_over_sample  = over_sampler.fit_resample( df_all_x_data, df_all_y_data)

    #print(df_all_data_input["stroke"].value_counts())
    #exit()
    # rebuild_data = x_all_data_under_sample
    # rebuild_data["stroke"] = y_all_data_under_sample
    # plot_correlation_data(rebuild_data, "Correlation of Balanced Data", "images/correlationOfBalancedData.jpg")
    # plot_correlation_data(df_all_data_input, "Data Correlation", "images/correlationOfALLData.jpg")
    # exit()

    ## Split Data
    x_train_split, x_test_split, y_train_split, y_test_split = train_test_split(df_all_x_data, df_all_y_data,
                                                                                test_size=0.4, random_state=0,
                                                                                stratify=df_all_y_data)
#
    ## Resample Training Data
    #x_train_over_sample, y_train_over_sample = over_sampler.fit_resample(x_train_split, y_train_split)
    x_train_under_sample, y_train_under_sample = under_sampler.fit_resample(x_train_split, y_train_split)
#
    ## Resample Test Data
    #x_test_over_sample,  y_test_over_sample  = over_sampler.fit_resample( x_test_split, y_test_split)
    x_test_under_sample, y_test_under_sample = under_sampler.fit_resample(x_test_split, y_test_split)

    #show_likelihood_of_stroke_if_ever_married(x_all_data_under_sample, y_all_data_under_sample)

    if resample_type_in == ResampleType.DISABLE:
        x_data_for_classifier = df_all_x_data
        y_data_for_classifier = df_all_y_data
    elif resample_type_in == ResampleType.OVER:
        x_data_for_classifier = x_all_data_over_sample
        y_data_for_classifier = y_all_data_over_sample
    elif resample_type_in == ResampleType.UNDER:
        x_data_for_classifier = x_all_data_under_sample
        y_data_for_classifier = y_all_data_under_sample
    else:
        x_data_for_classifier = x_all_data_under_sample
        y_data_for_classifier = y_all_data_under_sample

    classifier = None

    scaling = False
    scorer = ""
    classifier = get_classifier(test_algorithm)
    if test_algorithm == Algorithm.NAIVE_BAYES:
        # Instantiate the classifier
        scorer = "Gaussian Naive Bayes"
        #classifier = GaussianNB() #BernoulliNB()
        #classifier = BernoulliNB()
        scoring_function_min = score_single_dimension_test
    elif test_algorithm == Algorithm.LINEAR_SVM:
        scorer = "SVM Linear"
        #classifier = SVC(C=1, kernel='linear', gamma = 0.001)
        #scoring_function_min = score_two_dimension_test
        scoring_function_min = score_single_dimension_test
        scaling = True
    elif test_algorithm == Algorithm.LOGISTIC_REGRESSION:
        scorer = "Logistic Regression"
        #classifier = LogisticRegression(solver='liblinear', random_state=123)
        #scoring_function_min = score_two_dimension_test
        scoring_function_min = score_single_dimension_test
        scaling = False
    elif test_algorithm == Algorithm.SVM_RBF:
        scorer = "SVM RBF Kernel"
        #classifier = SVC(C=1, kernel='rbf', gamma=0.001)
        #classifier = SVC(C=1, kernel='rbf', gamma=0.1)
        #scoring_function_min = score_two_dimension_test
        scoring_function_min = score_single_dimension_test
    elif test_algorithm == Algorithm.DECISION_TREE:
        scorer = "Decision Tree"
        #classifier = DecisionTreeClassifier(random_state=123)
        #scoring_function_min = score_two_dimension_test
        scoring_function_min = score_single_dimension_test
    elif test_algorithm == Algorithm.K_NN:
        scorer = "KNN"
        #classifier = KNeighborsClassifier(n_neighbors=11, weights="distance")
        #scoring_function_min = score_two_dimension_test
        scoring_function_min = score_single_dimension_test
    else:
        scoring_function_min = score_single_dimension_test

    if use_minimal_variables_in:
        print("Using MINIMAL data in dataset")
        scoring_function = scoring_function_min
    else:
        print("Using ALL data in dataset")
        scoring_function = score_all_dimensions_test

    if classifier is not None:
        print(f"Testing {scorer}")
        score, best_columns = scoring_function(classifier, x_data_for_classifier, y_data_for_classifier, scaling)
        print(f"{scorer} Accuracy: {round(score, 5)}")

        print("Generating ROC Curve")
        plot_k_fold_roc_curve(classifier, x_data_for_classifier, y_data_for_classifier)

        if test_algorithm == Algorithm.LOGISTIC_REGRESSION:
            print("If you want to see the Regression Table. You must uncomment the lines below")
            #print("Showing Logistic Regression Table")
            #show_logistic_regression_table(x_all_data_under_sample, y_all_data_under_sample)

        if test_algorithm == Algorithm.K_NN:
            print("If you want to calculate optimal k uncomment lines below")
            #print("Finding optimal k between 1 and 20")
            #optimal_k = find_optimal_k_for_KNN(x_all_data_under_sample,y_all_data_under_sample)

        if test_algorithm == Algorithm.SVM_RBF:
            pass
            #x_train_data_cm = x_train_under_sample[["age","hypertension"]]
            #x_test_data_cm  = x_test_under_sample[["age","hypertension"]]
            #y_train_data_cm = y_train_under_sample
            #classifier.fit(x_train_data_cm, y_train_data_cm)
            #predictions = classifier.predict(x_test_data_cm)
            #accuracy = accuracy_score(y_test_under_sample, predictions)
            #cm = confusion_matrix(y_test_under_sample,predictions, labels=classifier.classes_)
            #disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classifier.classes_)
            #disp.plot()
            #plt.show()
            #print(f"Accuracy of SVM_RBF {accuracy}")

        if test_algorithm == Algorithm.LINEAR_SVM:
            print("If you want to see the SVM GRAPH. You must uncomment the lines below")
            #all_column_names = list(x_data_for_classifier.columns)
            #col_list = [all_column_names[best_columns[0]], all_column_names[best_columns[1]]]
            #x_data_linear_svc = pd.DataFrame(columns=col_list)
            #x_data_linear_svc = x_data_for_classifier[col_list].to_numpy()
            #y_data_linear_scv = y_data_for_classifier.to_numpy()
            ##print(x_data_linear_svc)
            ##print(type(x_data_linear_svc))
            ##print(x_data_linear_svc)
            ##print(y_data_linear_scv)
            #plot_title = f"Support Vector Machine"
            #plot_linear_svc(x_data_linear_svc, y_data_linear_scv, plot_title)
            #plt.xlabel(all_column_names[best_columns[0]])
            #plt.ylabel(all_column_names[best_columns[1]])
            #plt.savefig("images/SVM_Linear.jpg")
            #plt.show()
        print("")



if __name__ == "__main__":
    # Parameters
    use_minimal_variables = True
    resample_type         = ResampleType.UNDER
    input_file_stroke_data = "input/healthcare-dataset-stroke-data.csv"

    # Import Data
    df_all_data = pd.read_csv(input_file_stroke_data)

    # Clean Data
    df_all_data = clean_stroke_data(df_all_data)

    parametric_algorithm_list = [Algorithm.NAIVE_BAYES,
                                 Algorithm.LINEAR_SVM,
                                 Algorithm.LOGISTIC_REGRESSION]

    non_parametric_algorithm_list = [Algorithm.K_NN,
                                     Algorithm.DECISION_TREE,
                                     Algorithm.SVM_RBF]

    for algorithm in parametric_algorithm_list:
       test_algorithm_with_data(algorithm, df_all_data, use_minimal_variables, resample_type)

    for algorithm in non_parametric_algorithm_list:
        test_algorithm_with_data(algorithm, df_all_data, use_minimal_variables, resample_type)

    #best_algorithms_list = [Algorithm.LOGISTIC_REGRESSION, Algorithm.SVM_RBF]
    #for algorithm in best_algorithms_list:
    #    test_algorithm_with_data(algorithm, df_all_data, use_minimal_variables)

    #generate_mean_ROC_curve_for_classifiers([Algorithm.NAIVE_BAYES, Algorithm.SVM_RBF], df_all_data,"Best_algorithms_ROC_underSampling.jpg")

    df_x_data = df_all_data.iloc[:, :-1]
    df_x_data = df_x_data.drop(["gender", "Residence_type"], axis=1)
    df_y_data = df_all_data.iloc[:, -1]

    resample_type = ResampleType.UNDER
    plot_k_fold_roc_curve(Algorithm.SVM_RBF, df_x_data, df_y_data, resample_type)
    plot_k_fold_roc_curve(Algorithm.NAIVE_BAYES, df_x_data, df_y_data, resample_type)

