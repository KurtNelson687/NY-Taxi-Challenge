"""
Created on Mon Jan 22 17:07:41 2018
@author: kurtnelson

Summary: taxiRides.py was created as part of the kaggle data challenge "New York
City Taxi Trip Duration" (https://www.kaggle.com/c/nyc-taxi-trip-duration/data).
The taxiRides class and associated functions are used to analyzes taxi ride data 
and predict future ride durations. To create a taxiRide object, a csv file in the
format specified by the kaggle data challenge "New York City Taxi Trip Duration"
must be specified. Once created, a taxi taxiRides object stores data about 
individual trips including taxi vendor ID, passenger count, pickup and drop off
locations (latitude and longitude) and times, trip distance and duration, and
average speed. The taxiRides class also contians various plotting funciton used
to analyze and view the ride data, and to predict future rides. 
"""
        
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import xgboost as xgb
from geopy.distance import vincenty
from sklearn.cluster import KMeans

sns.set()

class taxiRides():
    
    def __init__(self,trainFile):
        """
        Function: __init__
        Purpose:  Create taxiRides object that stores data about individual 
        trips including taxi vendor ID, passenger count, pickup and drop off
        locations (latitude and longitude) and times, trip distance and
        duration, and average speed. 
        """
        secToHour = 1/60*1/60 # seconds to hour conversion
        
        self.trainData = pd.read_csv(trainFile) # load data into dataFrame
        self.trainData['trip_duration'] = self.trainData['trip_duration']*secToHour # convert duration to hours
        self.extractTimes() # extract time data
        self.computeTripDistances() # compute vincenty distance between pickup and dropoff
        # self.trainData = self.extractTimes() # extract time data
       # self.trainData = self.computeTripDistances() # compute vincenty distance between pickup and dropoff
        self.trainData['aveSpeed'] =  self.trainData[
                'p2pDistance']/self.trainData['trip_duration'] # Compute average trip speed in miles/hr
                

    def cleanTrainingData(self,qLow,qHigh):
        """
        Function: cleanData
        Purpose: removes outliers from data and plots before and after histograms of
        duration, distance, and average speed. Data outside the quantiles specifed by
        qLow and qHigh are removed.
        
        Inputs:
            1) qLow - lower quantile for data removal
            2) qHigh - higher quantile for data removal
    
        Output:   
            1) trainData - cleaned trainData
        """
        numStart = len(self.trainData) # starting number of elments in dataframe
        
        self.quantileClean('dropoff_latitude',qLow,qHigh)
        self.quantileClean('dropoff_longitude',qLow,qHigh)
        self.quantileClean('trip_duration',qLow,qHigh)
        self.quantileClean('p2pDistance',qLow,qHigh)
        self.quantileClean('aveSpeed',qLow,qHigh)
                    
        numEnd = len(self.trainData) # ending number of elments in dataframe
        
        print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
        print((numStart-numEnd)/numStart*100, '% of the elements were removed in data cleaning')
        print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
    
    def quantileClean(self,var,qLow,qHigh):
        """
        Function: quantileClean
        Purpose: Removes outliers based on upper and lower quantile.
        
        Inputs:
            1) var - variable to clean
            2) qHigh - upper quantile
            3) qLow - lower quantile
           
        Output:   
            1) trainData - with cleaned var
        """
        lowerLim = self.trainData[var].quantile(qLow) # lower limit
        upperLim = self.trainData[var].quantile(qHigh) # upper limit

        self.trainData = self.trainData.loc[(
                self.trainData[var] > lowerLim) & (self.trainData[var] < upperLim)]
        
    def plotHistograms(self):
        """
        Function: plotHistograms
        Purpose: Plot marginal probability mass functions for the log of trip duration,
        trip distance, and average speed.
        """
        plt.figure(figsize=(3.55, 3.55))
        # plot histrogram of trip duration
        plt.subplot(3, 1, 1)
        sns.distplot(np.log(self.trainData['trip_duration']+1), axlabel = 'ln(trip duration)',
                     bins = 50, norm_hist=True,color="b")
        
        # plot histrogram of distance
        plt.subplot(3, 1, 2)
        sns.distplot(np.log(self.trainData['p2pDistance']+1), axlabel = 'ln(trip distance)',
                     bins = 50, norm_hist=True,color="r")

        # plot histrogram average speed
        plt.subplot(3, 1, 3)
        sns.distplot(np.log(self.trainData['aveSpeed']+1), axlabel = 'ln(ave trip speed)',
                     bins = 50, norm_hist=True,color="g")
                    
    def extractTimes(self):
        """
        Function: extractTimes
        Purpose: Converts time to datatime and extracts month, day of week, and hour for 
        drop off and pick up times from dataFrame
        
        Output:   
            1) DropOffMonth - drop off month
            2) DropOffHour - drop off hour
            3) DropOffDayOfWeek - drop off day of week
            4) PickupMonth - pickup month
            5) PickupHour - pickup hour
            6) PickupDayOfWeek - pickup day of week
        """
        # Convert pickup and dropoff times to datetime
        self.trainData['dropoff_datetime'] = pd.to_datetime(self.trainData.dropoff_datetime)
        self.trainData['pickup_datetime'] = pd.to_datetime(self.trainData.pickup_datetime)

        # Extract month, hour, and day of week for dropoffs
        self.trainData['DropOffMonth'] = self.trainData.dropoff_datetime.dt.month # drop off month
        self.trainData['DropOffHour'] = self.trainData.dropoff_datetime.dt.hour # drop off hour
        self.trainData['DropOffDayOfWeek'] = self.trainData.dropoff_datetime.dt.dayofweek # drop off day of week
        
        # Extract month, hour, and day of week for pickups
        self.trainData['PickupMonth'] = self.trainData.pickup_datetime.dt.month # pickup month
        self.trainData['PickUpHour'] = self.trainData.pickup_datetime.dt.hour # pickup hour
        self.trainData['PickUpDayOfWeek'] = self.trainData.pickup_datetime.dt.dayofweek # pickup day of week
            
    def computeTripDistances(self):
        """
        Function: computeTripDistances
        Purpose: Pair latitude and longitude of drop off and pickup, then computes
        the distance between the two. 
        
        Output:   
            1) PickupLoc - paired lat and long for pickup
            2) DropOffLoc - paired lat and long for drop off
            3) p2pDistance - vincenty distance between pickup and drop off
        """        
        self.trainData['PickupLoc'] = list(zip(
                self.trainData['pickup_latitude'],self.trainData['pickup_longitude']))
        self.trainData['DropOffLoc'] = list(zip(
                self.trainData['dropoff_latitude'],self.trainData['dropoff_longitude']))
        self.trainData['p2pDistance'] = list(map(
                self.getDistance, self.trainData['PickupLoc'], self.trainData['DropOffLoc']))
        return self.trainData

        
    def getDistance(self,point1,point2):
        """
        Function: getDistance
        Purpose: Compute vincenty distance between two points in miles 
        Inputs:   
            1) point1 - tuple containing latitude and longitude of point 1
            2) point2 - tuple containing latitude and longitude of point 2
        
        Output:   
            1) vincenty distance in miles
        """
        return vincenty(point1,point2).miles
        
    def showTrainStats(self):
        """
        Function: showTrainStats
        Purpose:  Shows statistics for data in self.trainData
        Output:   
            1) Display of train data statistics 
        """
        print(self.trainData.describe())
     
    def makeBinTicks(self,varName,center):
        """
        Function: makeBinTicks
        Purpose:  Makes xtick labels and tick locations
        Inputs:   
            1) varName - name of plotted variable
            2) center - if true adjust location so label is
            centered for histograms and pcolor plots. 

        Output:   
            1) ticks - tick locations
            2) labels - tick labels        
        """
        var = self.trainData[varName] # data being plotted
        if varName in ['DropOffDayOfWeek', 'PickUpDayOfWeek']: # label for day of week
            labels = ['Mon', 'Tue', 'Wed', 'Thu','Fri','Sat','Sun']
            if center: # labels for histograms and pcolor
                ticks = np.linspace(min(var)+0.5, max(var)-0.5, num=7)
            else:
                ticks = np.linspace(min(var), max(var), num=7)
        elif varName in ['DropOffHour', 'PickUpHour']: # label for hour of day
            labels = ['12 AM', '3 AM', '6 AM', '9 AM','12 PM','3 PM', '6 PM', '9 PM']
            ticks = np.arange(0,24,3)    
        else: # labels for everything else
             labels = np.linspace(min(var), max(var), num=5)
             if center: # labels for histograms and pcolor
                ticks = np.linspace(min(var)+0.5, max(var)-0.5, num=5)        
             else:
                ticks = np.linspace(min(var), max(var), num=5)  
        
        return ticks, labels
        
    def showJointProbMass(self,var1,var2):
        """
        Function: showJointProbMass
        Purpose:  Plot the joint probabilty mass function of any two varibales in self.trainData
        Inputs:   
            1) var1 - independent variable on x-axis
            2) var2 - independent variable on y-axis 
        Output:   
            1) Plot of joint probability mass function
        """
        # Set variables
        x = self.trainData[var1]
        y = self.trainData[var2]

        # Create plot
        xedges = np.linspace(min(x), max(x), num=len(np.unique(x))+1) # x bin edges
        yedges = np.linspace(min(y), max(y), num=len(np.unique(y))+1) # y bin edges
        
        freq, monthEdges, dayEdges = np.histogram2d(x,y,bins =(xedges,yedges),normed=True) # create mass function
        freq = freq.T
        X, Y = np.meshgrid(xedges, yedges) # plotting grid
        plt.pcolormesh(X, Y, freq,cmap ='RdBu')
        cbar = plt.colorbar()
        cbar.set_label('probability')      
    
    def dailyAve(self, var,splitVendor):
        """
        Function: WeeklyTimeSeries
        Purpose:  Plot a time series of daily average of var. Options to plot
        individual series for each taxi vendor
            1) var - variable to plot
            2) splitVendor - flag to specify whether or not to split data by vendor.
            Set as True to split data.
        Output:   
            1) Time series plot
        """
        if splitVendor: # plot showing weekly average seperate for each vendor
            plotFrame = pd.DataFrame(self.trainData.groupby(['vendor_id','DropOffDayOfWeek'])[var].mean())
            plotFrame.reset_index(inplace = True)
            plotFrame['unit'] = 1
            sns.tsplot(data = plotFrame, time = 'DropOffDayOfWeek', unit = "unit",condition = "vendor_id", value = var)
        else:  # plot showing total weekly average
            plotFrame = pd.DataFrame(self.trainData.groupby(['DropOffDayOfWeek'])[var].mean())
            plotFrame.reset_index(inplace = True)
            plotFrame['unit'] = 1
            sns.tsplot(data = plotFrame, time = 'DropOffDayOfWeek', unit = "unit", value = var)
            
    def makeHeatMap(self,var):
        """
        Function: makeHeatMap
        Purpose: Make heatmap based on time of day and day of week
            1) var - variable to plot
        Output:   
            1) Heatmap plot of daily and hourly mean of var
        """
        # group by drop off day of week and drop off hour
        plotFrame = pd.DataFrame(self.trainData.groupby(['DropOffDayOfWeek','DropOffHour'])[var].mean())        
        plotFrame = plotFrame.unstack(0) # remove one of the row multiindices
        plotFrame = plotFrame[var] # extract averaged data
        ax = sns.heatmap(plotFrame,cmap='RdBu_r') # plot heatmap
        ax.invert_yaxis() # flip y-axis
        
    def computeCluster(self,numK):
        """
        Function: computeCluster
        Purpose: Compute, plot, and save cluster centroids using k-means clustering. 
        If locations are spread out, this algorithm should not be applied, because it 
        is based on the eculdian between latitude and longitude coordinates, and hence
        does not account for earths curvature.
            1) numK - number of clusters for k-means
        Output:   
            2) kmeanFit - kmeans model object from fit
        """        
        loc_df = pd.DataFrame() # create data frame storing all dropoff and pickup locations
        loc_df['longitude'] = list(self.trainData[
                'dropoff_longitude']) + list(self.trainData['pickup_longitude'])
        loc_df['latitude'] = list(self.trainData[
                'dropoff_latitude']) + list(self.trainData['pickup_latitude'])
        
        # find clusters using k-means
        kmeanFit = KMeans(n_clusters=numK, n_init = 10).fit(loc_df)
     
        loc_df['centriodLabel'] = kmeanFit.labels_ # add cluster label to each point
     
        plt.figure(figsize = (3.55,3.55))
        colorInd = np.linspace(0,1,numK)

        #plot pickup and dropoff locations
        for ind,clusterNum in enumerate(loc_df.centriodLabel.unique()):
            # plot pickup and dropoff locations =
            plt.plot(loc_df.longitude[loc_df.centriodLabel==clusterNum],loc_df.latitude[
                    loc_df.centriodLabel==clusterNum], '.',markersize = 0.5,alpha=0.5, 
                     color=plt.cm.tab20(colorInd[ind]),rasterized=True)
        
        ax = plt.gca()

        # add cluster centriods    
        for ind,clusterNum in enumerate(loc_df.centriodLabel.unique()):
            plt.plot(kmeanFit.cluster_centers_[ind,0], kmeanFit.cluster_centers_[ind,1],
                     marker='.', markersize=5, color = 'black') # plot centriods
            ax.annotate(clusterNum, (kmeanFit.cluster_centers_[ind,0], 
                     kmeanFit.cluster_centers_[ind,1]),fontsize=8) # add text label to centriods
            
        plt.xlim([-74.05, -73.75])
        plt.ylim([40.6, 40.9])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_facecolor ("white")
        
        # save pickup and dropoff clusters
        self.trainData['DropoffCluster'] = np.array(loc_df.centriodLabel[0:len(self.trainData)])
        self.trainData['PickupCluster'] = np.array(loc_df.centriodLabel[len(self.trainData):None])

        return kmeanFit.cluster_centers_
    
    def rideCountAtClusters(self, clusters,binType):
        """
        Function: clusterCount
        Purpose:  Plot a time series of ride counts at each cluster, either daily or hourly. 
        Counts can based on drop off, pickup, or both drop off and pickup locations
            1) clusters - cluster assignment to count
            2) binType - type of time data to bin by
            Set as True to split data.
        Output:   
            1) Time series plot
        """
        plotFrame = pd.DataFrame(self.trainData.groupby([clusters,binType])[clusters].count())
        plotFrame = plotFrame.rename(columns={clusters: 'count'})
        plotFrame.reset_index(inplace = True)
        plotFrame['unit'] = 1
        sns.tsplot(data = plotFrame, time = binType, unit = "unit",condition = clusters, value = 'count')
        
    def getXGBoostTrain(self):
        """
        Function: arrangeXGBoostData
        Purpose: Prepare training data for XGBoost
        Output:   
            1) self.XGBtrain - traning data for XGBoost
        """
        duration_train = self.trainData['trip_duration']
        speed_trian = self.trainData['aveSpeed']
        distance_train = self.trainData['p2pDistance']
        ID_train = pd.get_dummies(self.trainData['vendor_id'], prefix='vi', prefix_sep='_')
        #passenger_count_train = pd.get_dummies(self.trainData['passenger_count'], prefix='pc', prefix_sep='_')
        #cluster_pick_train = pd.get_dummies(self.trainData['pickup_cluster'], prefix='p', prefix_sep='_')
        #cluster_drop_train = pd.get_dummies(self.trainData['dropoff_cluster'], prefix='d', prefix_sep='_')
        month_train = pd.get_dummies(self.trainData['DropOffMonth'], prefix='m', prefix_sep='_')
        hour_train = pd.get_dummies(self.trainData['DropOffHour'], prefix='h', prefix_sep='_')
        day_train = pd.get_dummies(self.trainData['DropOffDayOfWeek'], prefix='dow', prefix_sep='_')
        
        #XGB_train = pd.DataFrame() # crete training dataframe for XGBoost
        self.XGB_train = pd.concat([duration_train,
                          speed_trian,
                          distance_train,
                          ID_train,
                          month_train,
                          hour_train,
                         day_train
                         ], axis=1)
               
        print(self.XGB_train.describe())

    
    def learnXGBoost(self,params):
        """
        Function: learnXGBoost
        Purpose: Train an XGBoost model with the training data
            1) params - model paramters for XGBoost
        Output:   
            1) self.XGmodel - earned model
        """
        pass
        
    
if __name__ == "__main__":
     numK = 16 # number of clusters for K-mean
     NYtaxi = taxiRides("../dataFiles/train.csv")
     
     #######################################################
     # Remove outliers and show data statistics
     #######################################################
     #NYtaxi.cleanTrainingData(40.4,40.93,-74.1,-73.4,12.0,600.0,80.0) # clean traning data     
     NYtaxi.cleanTrainingData(0.009,0.999) # clean traning data     
     NYtaxi.showTrainStats() # show data statistics
     
     #######################################################
     # Plot histograms of trip speed, duration, and length
     #######################################################
     NYtaxi.plotHistograms()
     plt.tight_layout()
     plt.savefig('./Figures/marginalMasFunctions.pdf', format='pdf', dpi=1000)
     
     #######################################################
     # Plot weekly averages of trip duration, speed, and distance
     #######################################################
     plt.figure(figsize=(3.55, 4))
     ax = plt.subplot(3, 1, 1)
     NYtaxi.dailyAve('trip_duration',True)
     ax.set_xticklabels([])
     ax.set_xlabel('')
     ax.set_ylabel('Duration')
     ax.legend(loc='upper right')

     ax = plt.subplot(3, 1, 2)
     NYtaxi.dailyAve('p2pDistance',True)
     ax.set_xticklabels([])
     ax.set_xlabel('')
     ax.set_ylabel('Distance')
     ax.legend_.remove()

     ax = plt.subplot(3, 1, 3)
     NYtaxi.dailyAve('aveSpeed',True)
     xticks, xlabels = NYtaxi.makeBinTicks('DropOffDayOfWeek',False)
     ax.set_ylabel('Mean Speed')
     ax.set_xlabel('Day')
     plt.xticks(xticks, xlabels, rotation='vertical')
     ax.legend_.remove()

     plt.tight_layout()
     plt.savefig('./Figures/TimeSeries.pdf', format='pdf', dpi=1000)
     
     #######################################################
     # plot joint probability mass function  of drop off day
     # of week and hour
     #######################################################
     plt.figure(figsize=(3.55, 4))
     NYtaxi.showJointProbMass('DropOffDayOfWeek','DropOffHour')
     plt.tight_layout()
     ax = plt.gca()
     # y axis
     yticks, ylabels = NYtaxi.makeBinTicks('DropOffHour',True)
     plt.yticks(yticks, ylabels)      
     ax.set_ylabel('Drop-off Hour')
     
     # x-axis
     xticks, xlabels = NYtaxi.makeBinTicks('DropOffDayOfWeek',True)
     plt.xticks(xticks, xlabels, rotation='vertical')
     plt.xticks(xticks, xlabels)      
     ax.set_xlabel('Day')
     
     plt.tight_layout()
     plt.savefig('./Figures/JointMassFunction.pdf', format='pdf', dpi=1000)
     
     #######################################################
     # plot heatmaps of ride distance, duration, and speed
     #######################################################
     plt.figure(figsize=(8, 4))
     # plot trip distance
     ax = plt.subplot(1, 3, 1)
     ax.set_title("Average distance")
     NYtaxi.makeHeatMap('p2pDistance')
     #y-axis
     yticks, ylabels = NYtaxi.makeBinTicks('DropOffHour',True)
     ax.set_ylabel('Hour')
     #x-axis
     xticks, xlabels = NYtaxi.makeBinTicks('DropOffDayOfWeek',False)
     xticks = xticks+0.5
     plt.xticks(xticks, xlabels, rotation='vertical')
     ax.set_xlabel('')

     # plot trip duration
     ax = plt.subplot(1, 3, 2)
     ax.set_title("Average duration")
     NYtaxi.makeHeatMap('trip_duration')
     #y-axis
     ax.set_ylabel('')
     ax.set_yticklabels([])
     #x-axis
     plt.xticks(xticks, xlabels, rotation='vertical')

     # plot average speed
     ax = plt.subplot(1, 3, 3)
     NYtaxi.makeHeatMap('aveSpeed')
     ax.set_title("Average speed")
     #y-axis
     ax.set_ylabel('')
     ax.set_yticklabels([])
     #x-axis
     plt.xticks(xticks, xlabels, rotation='vertical')
     ax.set_xlabel('')

     plt.tight_layout()
     plt.savefig('./Figures/heatMapDuration.pdf', format='pdf', dpi=1000)
     
     #######################################################
     # compute clusters using k-means and plotted them
     #######################################################
     kCenters = NYtaxi.computeCluster(numK)
     plt.tight_layout()
     plt.savefig('./Figures/kMeansClusters.pdf', format='pdf',dpi = 1000)
     
     #######################################################
     # Plot weekly pickup and drop off counts at each cluster
     #######################################################
     plt.figure(figsize=(3.55, 4))
     ax = plt.subplot(2, 1, 1)
     NYtaxi.rideCountAtClusters('PickupCluster','PickUpDayOfWeek')
     ax.set_ylabel('# of Pickups')
     ax.set_xticklabels([])
     ax.set_xlabel('')
     # Shrink current axis and move legend
     #box = ax.get_position()
     #ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
     #ax.legend(loc='upper left', bbox_to_anchor=(1, 0.5))
     ax.legend_.remove()

     ax = plt.subplot(2, 1, 2)
     NYtaxi.rideCountAtClusters('DropoffCluster','DropOffDayOfWeek')
     ax.set_ylabel('# of Drop-offs')
     ax.set_xlabel('Day')
     #box = ax.get_position()
     #ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
     xticks, xlabels = NYtaxi.makeBinTicks('DropOffDayOfWeek',False)
     plt.xticks(xticks, xlabels, rotation='vertical')
     ax.legend_.remove()

     plt.tight_layout()
     plt.savefig('./Figures/ClusterCountDaily.pdf', format='pdf', dpi=1000)
     
     #######################################################
     # Plot hourly pickup and drop off counts at each cluster
     #######################################################
     plt.figure(figsize=(3.55, 4)) 
     
     ax = plt.subplot(2, 1, 1) # pickup plot
     NYtaxi.rideCountAtClusters('PickupCluster','PickUpHour')
     ax.set_ylabel('# of Pickups')
     ax.set_xticklabels([])
     ax.set_xlabel('')
     # Shrink current axis and move legend
     #box = ax.get_position()
     #ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
     #ax.legend(loc='upper left', bbox_to_anchor=(1, 0.5))
     ax.legend_.remove()

     ax = plt.subplot(2, 1, 2) #drop off plot
     NYtaxi.rideCountAtClusters('DropoffCluster','DropOffHour')
     ax.set_ylabel('# of Drop-offs')
     ax.set_xlabel('Hour')
     #box = ax.get_position()
     #ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
     xticks, xlabels = NYtaxi.makeBinTicks('DropOffHour',False)
     plt.xticks(xticks, xlabels, rotation='vertical')
     ax.legend_.remove()

     plt.tight_layout()
     plt.savefig('./Figures/ClusterCountHour.pdf', format='pdf', dpi=1000)
     
     #######################################################
     # plot joint probability mass function  of drop off day
     # of week and hour
     #######################################################
     plt.figure(figsize=(3.55, 3.3))
     NYtaxi.showJointProbMass('PickupCluster','DropoffCluster')
     plt.tight_layout()
     ax = plt.gca()
     
     labels = np.array([0,3,6,9,12,15])
     ticks = np.linspace(0.5,14.5,6)
     
     # y-axis
     plt.yticks(ticks, labels)      
     ax.set_ylabel('Pickup Cluster')
     # x-axis
     xticks, xlabels = NYtaxi.makeBinTicks('PickupCluster',True)
     plt.xticks(ticks, labels)
     ax.set_xlabel('Drop-off Cluster')
     
     plt.tight_layout()
     plt.savefig('./Figures/JointMassCluster.pdf', format='pdf', dpi=1000)
     
     #######################################################
     # Fit XGBoost
     #######################################################
     NYtaxi.getXGBoostTrain()
    # xgb_pars = {'min_child_weight': 1, 'eta': 0.45, 'colsample_bytree': 0.8, 
    #             'max_depth': 6,'subsample': 0.8, 'lambda': 1., 'nthread': -1,
    #             'booster' : 'gbtree', 'silent': 1,'eval_metric': 'rmse', 
    #             'objective': 'reg:linear'}
     
    # model = xgb.train(xgb_pars, dtrain, 10, watchlist, early_stopping_rounds=2,
     #                  maximize=False, verbose_eval=1)
    # print('Modeling RMSLE %.5f' % model.best_score)