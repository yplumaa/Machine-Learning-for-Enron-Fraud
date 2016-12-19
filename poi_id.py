#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
#features_list = ['poi','salary'] # You will need to use more features
features_list=['poi','bonus', 'salary','exercised_stock_options','total_stock_value',\
               'deferred_income','long_term_incentive', 'restricted_stock',\
               'from_poi_to_this_person', 'to_poi_fraction','shared_receipt_with_poi']

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
data_dict.pop('TOTAL',0)

### Task 3: Create new feature(s)
def computeFraction( poi_messages, all_messages ):
    fraction = 0.    
    if poi_messages != "NaN" and all_messages != "NaN":
        fraction= 1.0*poi_messages/all_messages
    return fraction
for name in data_dict:

    data_point = data_dict[name]

    from_poi_to_this_person = data_point["from_poi_to_this_person"]
    to_messages = data_point["to_messages"]
    fraction_from_poi = computeFraction( from_poi_to_this_person, to_messages )


    from_this_person_to_poi = data_point["from_this_person_to_poi"]
    from_messages = data_point["from_messages"]
    fraction_to_poi = computeFraction( from_this_person_to_poi, from_messages )
    
    shared_receipt_with_poi = data_point["shared_receipt_with_poi"]
    to_messages = data_point["to_messages"]
    fraction_share_poi = computeFraction( shared_receipt_with_poi, from_messages )
    
    data_dict[name].update({"from_poi_to_fraction":fraction_from_poi,
                       "to_poi_fraction":fraction_to_poi,
                        "share_receipt_poi_fraction":fraction_share_poi})

    ### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
clf=Pipeline([('scaler',MinMaxScaler()),\
                ('pca', PCA(n_components=5,copy=True, whiten=False,random_state=42)),
                ('clfNB', GaussianNB())])
# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import KFold
kf=KFold(len(labels),4,shuffle=True)
for train_indeces, test_indeces in kf:
    features_train=[features[ii] for ii in train_indeces]
    features_test=[features[ii] for ii in test_indeces]
    labels_train=[labels[ii] for ii in train_indeces]
    labels_test=[labels[ii] for ii in test_indeces]
    clf.fit(features_train, labels_train)
    pred=clf.predict(features_test)
    from sklearn.metrics import accuracy_score
    print "accuracy:",accuracy_score(pred, labels_test)
    from sklearn.metrics import classification_report
    print classification_report(labels_test, pred)
### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.
dump_classifier_and_data(clf, my_dataset, features_list)