#Employee attrition
#objectivve : automate the process of predicting employee's attrition

#data understanding
#loading libraries
library(plyr)
library(ggplot2)
library(cowplot)

library(mltools)
library(MASS)
library(car)
library(GGally)
library(e1071)

library(ROCR)
library(dplyr)
library(caTools)
library(caret)

#load data files
emp_survey <- read.csv("employee_survey_data.csv",stringsAsFactors = FALSE)
mngr_survey <- read.csv("manager_survey_data.csv",stringsAsFactors = FALSE)
emp_info <- read.csv("general_data.csv",stringsAsFactors = FALSE)
in_time <- read.csv("in_time.csv",stringsAsFactors = FALSE)
out_time <- read.csv("out_time.csv",stringsAsFactors = FALSE)

#brief summary of files

str(emp_survey)   
str(mngr_survey)  
str(in_time)      
str(out_time) 

#data preparation
#prepare in time and out time
in_time[2:262] <- sapply(in_time[2:262], function(x) round((as.numeric(format(as.POSIXct(x,format="%Y-%m-%d %H:%M:%S"),"%H")) + as.numeric(format(as.POSIXct(x,format="%Y-%m-%d %H:%M:%S"),"%M"))/60),2))
out_time[2:262] <- sapply(out_time[2:262], function(x) round((as.numeric(format(as.POSIXct(x,format="%Y-%m-%d %H:%M:%S"),"%H")) + as.numeric(format(as.POSIXct(x,format="%Y-%m-%d %H:%M:%S"),"%M"))/60),2))

#data frame for storing number of hours
total_time <- out_time
total_time[2:262] <- out_time[2:262]-in_time[2:262]

#avergae hours worked by employee
avg_hrs_worked <- vector('numeric')
for (i in 1:4410){
  avg_hrs_worked[i] <- round(sum(total_time[i,2:262],na.rm = TRUE)/12,2)
}

emp_avg_hrs <- cbind(total_time[1],avg_hrs_worked)
colnames(emp_avg_hrs)[1] <-"EmployeeID"


#Merge data in single file
length(unique(emp_info$EmployeeID))      # 4410 observations
length(unique(emp_survey$EmployeeID))    # 4410 observations
length(unique(mngr_survey$EmployeeID))   # 4410 observations
length(unique(emp_avg_hrs$EmployeeID))   # 4410 observations

setdiff(emp_info$EmployeeID,emp_survey$EmployeeID)  # Identical employeeID across these datasets
setdiff(emp_info$EmployeeID, mngr_survey$EmployeeID) # Identical employeeID across these datasets
setdiff(emp_info$EmployeeID, emp_avg_hrs$EmployeeID) # Identical employeeID across these datasets


employee <- join_all(list(emp_info,emp_survey,mngr_survey,emp_avg_hrs), by ="EmployeeID",type = "inner")

#employee has become the master file

View(employee)

str(employee)


#additional data prep
#columns with NA values
 
employee$JobSatisfaction[which(is.na(employee$JobSatisfaction))] <- 0
employee$WorkLifeBalance[which(is.na(employee$WorkLifeBalance))] <- 0
employee$EnvironmentSatisfaction[which(is.na(employee$EnvironmentSatisfaction))] <- 0

#replace NA in total working years and Num of companies worked
employee$NumCompaniesWorked <- ifelse(is.na(employee$NumCompaniesWorked) & (employee$TotalWorkingYears-employee$YearsAtCompany) == 0,1,
                                      ifelse(is.na(employee$NumCompaniesWorked) & (employee$TotalWorkingYears-employee$YearsAtCompany) ==1,0,employee$NumCompaniesWorked))

employee$TotalWorkingYears <- ifelse(is.na(employee$TotalWorkingYears) & (employee$NumCompaniesWorked+employee$YearsAtCompany) == (employee$YearsAtCompany+1),employee$YearsAtCompany,
                                     ifelse(is.na(employee$TotalWorkingYears) & (employee$NumCompaniesWorked+employee$YearsAtCompany) ==(employee$YearsAtCompany),0,employee$TotalWorkingYears))


#treatment of outliers
employee$avg_hrs_worked[which(employee$avg_hrs_worked>213.7784)]<-213.7784
employee$TrainingTimesLastYear[which(employee$TrainingTimesLastYear>4)]<-4
employee$TrainingTimesLastYear[which(employee$TrainingTimesLastYear<1)]<-1
employee$YearsSinceLastPromotion[which(employee$YearsSinceLastPromotion>9)]<-9
employee$YearsWithCurrManager[which(employee$YearsWithCurrManager>14)]<-14
employee$MonthlyIncome[which(employee$MonthlyIncome>137756.0)]<-137756.0
employee$TotalWorkingYears[which(employee$TotalWorkingYears>32)]<-32
employee$YearsAtCompany[which(employee$YearsAtCompany>24)]<-24

#treating missing values
sapply(employee, function(x) sum(is.na(x)))
View(subset(employee, is.na(NumCompaniesWorked)))
View(subset(employee, is.na(TotalWorkingYears)))

employee <- employee[!is.na(employee$NumCompaniesWorked),]
employee <- employee[!is.na(employee$TotalWorkingYears),]
View(employee)

#correcting format for variables

employee$WorkLifeBalance <- mapvalues(employee$WorkLifeBalance, from = c(0,1,2,3,4), to = c("No response","Bad","Good","Better","Best"))
employee$JobInvolvement <- mapvalues(employee$JobInvolvement, from = c(1,2,3,4), to = c("Low","Medium","High","Very High"))
employee$PerformanceRating <- mapvalues(employee$PerformanceRating, from = c(3,4), to = c("Excellent","Outstanding"))
employee$Education <- mapvalues(employee$Education, from = c(1,2,3,4,5), to = c("Below College","College","Bachelor","Master","Doctor"))
employee$EnvironmentSatisfaction <- mapvalues(employee$EnvironmentSatisfaction, from = c(0,1,2,3,4), to = c("No response","Low","Medium","High","Very High"))
employee$JobSatisfaction <- mapvalues(employee$JobSatisfaction, from = c(0,1,2,3,4), to = c("No response","Low","Medium","High","Very High"))


#standardisation
#normalise features

employee$Age<- scale(employee$Age) 
employee$DistanceFromHome<- scale(employee$DistanceFromHome) 
employee$MonthlyIncome<- scale(employee$MonthlyIncome) 
employee$NumCompaniesWorked<- scale(employee$NumCompaniesWorked) 
employee$PercentSalaryHike<- scale(employee$PercentSalaryHike) 
employee$TotalWorkingYears<- scale(employee$TotalWorkingYears)
employee$TrainingTimesLastYear<- scale(employee$TrainingTimesLastYear) 
employee$YearsAtCompany<- scale(employee$YearsAtCompany) 
employee$YearsSinceLastPromotion<- scale(employee$YearsSinceLastPromotion) 
employee$YearsWithCurrManager<- scale(employee$YearsWithCurrManager) 
employee$avg_hrs_worked<- scale(employee$avg_hrs_worked) 

#convert yes no to 0 and 1 levels
employee$Attrition<- ifelse(employee$Attrition=="Yes",1,0)

#verify/check attrition rate
attrition <- sum(employee$Attrition)/nrow(employee)
attrition

#df of catagorical features
employee_chr<- employee[,-c(1,2,5,8,9,14,15,16,17,18,20,21,22,23,24,30)]

#convert cat variables into factors
employee_fact<- data.frame(sapply(employee_chr, function(x) factor(x)))
str(employee_fact) 

#dummy variables
dummies<- data.frame(sapply(employee_fact, function(x) data.frame(model.matrix(~x-1,data =employee_fact))[,-1]))
View(dummies)

#final data set
employee_final<- cbind(employee[,-c(3,4,6:13,16,18,19,25:29)],dummies)
employee_final <- employee_final[,-12]
View(employee_final)
str(employee_final)



#--Splitting data into training and testing datasets--#

set.seed(100)
indices = sample.split(employee_final$Attrition, SplitRatio = 0.7)
train = employee_final[indices,]
test = employee_final[!(indices),]
View(test)


#-------------LOGISTIC REGRESSION MODELLING----------#
model_1 <- glm(Attrition~., data=employee_final, family= "binomial")
summary(model_1) 

#--using STEPAIC function---#
library("MASS")
model_2 <- stepAIC(model_1, direction = "both")
summary(model_2) #AIC: 3206.8

#--Removing variable with multicollinearity---#
vif(model_2)

# removing EducationField.xLife.Sciences due to high vif of 15 and p value higher than 0.067745 #
model_3 <- glm(formula = Attrition ~ Age + NumCompaniesWorked + PercentSalaryHike + 
                 TotalWorkingYears + TrainingTimesLastYear + YearsAtCompany + 
                 YearsSinceLastPromotion + YearsWithCurrManager + BusinessTravel.xTravel_Frequently + 
                 BusinessTravel.xTravel_Rarely + Department.xResearch...Development + 
                 Department.xSales + Education.xCollege + Education.xDoctor + 
                 EducationField.xMarketing + 
                 EducationField.xMedical + EducationField.xOther + EducationField.xTechnical.Degree + 
                 JobLevel.x5 + JobRole.xLaboratory.Technician + JobRole.xManufacturing.Director + 
                 JobRole.xResearch.Director + JobRole.xResearch.Scientist + 
                 JobRole.xSales.Executive + MaritalStatus.xMarried + MaritalStatus.xSingle + 
                 StockOptionLevel.x1 + EnvironmentSatisfaction.xLow + EnvironmentSatisfaction.xVery.High + 
                 JobSatisfaction.xLow + JobSatisfaction.xVery.High + WorkLifeBalance.xBest + 
                 WorkLifeBalance.xBetter + WorkLifeBalance.xGood + WorkLifeBalance.xNo.response + 
                 JobInvolvement.xLow + JobInvolvement.xMedium + JobInvolvement.xVery.High, 
               family = "binomial", data = employee_final)
summary(model_3)
vif(model_3) # BusinessTravel.xTravel_Frequently, BusinessTravel.xTravel_Rarely, Department.xResearch...Development, Department.xSales have vif around 4, however they are highly significant and therefroe cannot be removed from the model#

#---elimminating variables with low significance of p value #

#EducationField.xMarketing has highest p value, so is removed in model_4#

model_4 <- glm(formula = Attrition ~ Age + NumCompaniesWorked + PercentSalaryHike + 
                 TotalWorkingYears + TrainingTimesLastYear + YearsAtCompany + 
                 YearsSinceLastPromotion + YearsWithCurrManager + BusinessTravel.xTravel_Frequently + 
                 BusinessTravel.xTravel_Rarely + Department.xResearch...Development + 
                 Department.xSales + Education.xCollege + Education.xDoctor + 
                 EducationField.xMedical + EducationField.xOther + EducationField.xTechnical.Degree + 
                 JobLevel.x5 + JobRole.xLaboratory.Technician + JobRole.xManufacturing.Director + 
                 JobRole.xResearch.Director + JobRole.xResearch.Scientist + 
                 JobRole.xSales.Executive + MaritalStatus.xMarried + MaritalStatus.xSingle + 
                 StockOptionLevel.x1 + EnvironmentSatisfaction.xLow + EnvironmentSatisfaction.xVery.High + 
                 JobSatisfaction.xLow + JobSatisfaction.xVery.High + WorkLifeBalance.xBest + 
                 WorkLifeBalance.xBetter + WorkLifeBalance.xGood + WorkLifeBalance.xNo.response + 
                 JobInvolvement.xLow + JobInvolvement.xMedium + JobInvolvement.xVery.High, 
               family = "binomial", data = employee_final)
summary(model_4)
vif(model_4)

#EducationField.xTechnical.Degree has highest p value, so is removed in model_5#

model_5 <- glm(formula = Attrition ~ Age + NumCompaniesWorked + PercentSalaryHike + 
                 TotalWorkingYears + TrainingTimesLastYear + YearsAtCompany + 
                 YearsSinceLastPromotion + YearsWithCurrManager + BusinessTravel.xTravel_Frequently + 
                 BusinessTravel.xTravel_Rarely + Department.xResearch...Development + 
                 Department.xSales + Education.xCollege + Education.xDoctor +
                 EducationField.xMedical + EducationField.xOther +  
                 JobLevel.x5 + JobRole.xLaboratory.Technician + JobRole.xManufacturing.Director + 
                 JobRole.xResearch.Director + JobRole.xResearch.Scientist + 
                 JobRole.xSales.Executive + MaritalStatus.xMarried + MaritalStatus.xSingle + 
                 StockOptionLevel.x1 + EnvironmentSatisfaction.xLow + EnvironmentSatisfaction.xVery.High + 
                 JobSatisfaction.xLow + JobSatisfaction.xVery.High + WorkLifeBalance.xBest + 
                 WorkLifeBalance.xBetter + WorkLifeBalance.xGood + WorkLifeBalance.xNo.response + 
                 JobInvolvement.xLow + JobInvolvement.xMedium + JobInvolvement.xVery.High, 
               family = "binomial", data = employee_final)
summary(model_5)
vif(model_5)

#EducationField.xMedical has highest p value, so is removed in model_6#

model_6 <- glm(formula = Attrition ~ Age + NumCompaniesWorked + PercentSalaryHike + 
                 TotalWorkingYears + TrainingTimesLastYear + YearsAtCompany + 
                 YearsSinceLastPromotion + YearsWithCurrManager + BusinessTravel.xTravel_Frequently + 
                 BusinessTravel.xTravel_Rarely + Department.xResearch...Development + 
                 Department.xSales + Education.xCollege + Education.xDoctor +
                 EducationField.xOther +  
                 JobLevel.x5 + JobRole.xLaboratory.Technician + JobRole.xManufacturing.Director + 
                 JobRole.xResearch.Director + JobRole.xResearch.Scientist + 
                 JobRole.xSales.Executive + MaritalStatus.xMarried + MaritalStatus.xSingle + 
                 StockOptionLevel.x1 + EnvironmentSatisfaction.xLow + EnvironmentSatisfaction.xVery.High + 
                 JobSatisfaction.xLow + JobSatisfaction.xVery.High + WorkLifeBalance.xBest + 
                 WorkLifeBalance.xBetter + WorkLifeBalance.xGood + WorkLifeBalance.xNo.response + 
                 JobInvolvement.xLow + JobInvolvement.xMedium + JobInvolvement.xVery.High, 
               family = "binomial", data = employee_final)
summary(model_6)
vif(model_6)

# YearsAtCompany has highest p value, so is removed in model_7#
model_7 <-glm(formula = Attrition ~ Age + NumCompaniesWorked + PercentSalaryHike + 
                TotalWorkingYears + TrainingTimesLastYear +  
                YearsSinceLastPromotion + YearsWithCurrManager + BusinessTravel.xTravel_Frequently + 
                BusinessTravel.xTravel_Rarely + Department.xResearch...Development + 
                Department.xSales + Education.xCollege + Education.xDoctor +
                EducationField.xOther +  
                JobLevel.x5 + JobRole.xLaboratory.Technician + JobRole.xManufacturing.Director + 
                JobRole.xResearch.Director + JobRole.xResearch.Scientist + 
                JobRole.xSales.Executive + MaritalStatus.xMarried + MaritalStatus.xSingle + 
                StockOptionLevel.x1 + EnvironmentSatisfaction.xLow + EnvironmentSatisfaction.xVery.High + 
                JobSatisfaction.xLow + JobSatisfaction.xVery.High + WorkLifeBalance.xBest + 
                WorkLifeBalance.xBetter + WorkLifeBalance.xGood + WorkLifeBalance.xNo.response + 
                JobInvolvement.xLow + JobInvolvement.xMedium + JobInvolvement.xVery.High, 
              family = "binomial", data = employee_final)
summary(model_7)
vif(model_7)

# BusinessTravel.xTravel_Rarely  has comparitiely high  p value, so is removed in model_8#
model_8 <- glm(formula = Attrition ~ Age + NumCompaniesWorked + PercentSalaryHike + 
                 TotalWorkingYears + TrainingTimesLastYear +  
                 YearsSinceLastPromotion + YearsWithCurrManager + BusinessTravel.xTravel_Frequently + 
                 Department.xResearch...Development + 
                 Department.xSales + Education.xCollege + Education.xDoctor +
                 EducationField.xOther +  
                 JobLevel.x5 + JobRole.xLaboratory.Technician + JobRole.xManufacturing.Director + 
                 JobRole.xResearch.Director + JobRole.xResearch.Scientist + 
                 JobRole.xSales.Executive + MaritalStatus.xMarried + MaritalStatus.xSingle + 
                 StockOptionLevel.x1 + EnvironmentSatisfaction.xLow + EnvironmentSatisfaction.xVery.High + 
                 JobSatisfaction.xLow + JobSatisfaction.xVery.High + WorkLifeBalance.xBest + 
                 WorkLifeBalance.xBetter + WorkLifeBalance.xGood + WorkLifeBalance.xNo.response + 
                 JobInvolvement.xLow + JobInvolvement.xMedium + JobInvolvement.xVery.High, 
               family = "binomial", data = employee_final)
summary(model_8)
vif(model_8)

# MaritalStatus.xMarried has non significant p value, so is removed in model_9#
model_9 <- glm(formula = Attrition ~ Age + NumCompaniesWorked + PercentSalaryHike + 
                 TotalWorkingYears + TrainingTimesLastYear +  
                 YearsSinceLastPromotion + YearsWithCurrManager + BusinessTravel.xTravel_Frequently + 
                 Department.xResearch...Development + 
                 Department.xSales + Education.xCollege + Education.xDoctor +
                 EducationField.xOther +  
                 JobLevel.x5 + JobRole.xLaboratory.Technician + JobRole.xManufacturing.Director + 
                 JobRole.xResearch.Director + JobRole.xResearch.Scientist + 
                 JobRole.xSales.Executive + MaritalStatus.xSingle + 
                 StockOptionLevel.x1 + EnvironmentSatisfaction.xLow + EnvironmentSatisfaction.xVery.High + 
                 JobSatisfaction.xLow + JobSatisfaction.xVery.High + WorkLifeBalance.xBest + 
                 WorkLifeBalance.xBetter + WorkLifeBalance.xGood + WorkLifeBalance.xNo.response + 
                 JobInvolvement.xLow + JobInvolvement.xMedium + JobInvolvement.xVery.High, 
               family = "binomial", data = employee_final)
summary(model_9)
vif(model_9)

# PercentSalaryHike has non significant  value, so is removed in model_10#
model_10 <- glm(formula = Attrition ~ Age + NumCompaniesWorked + 
                  TotalWorkingYears + TrainingTimesLastYear +  
                  YearsSinceLastPromotion + YearsWithCurrManager + BusinessTravel.xTravel_Frequently + 
                  Department.xResearch...Development + 
                  Department.xSales + Education.xCollege + Education.xDoctor +
                  EducationField.xOther +  
                  JobLevel.x5 + JobRole.xLaboratory.Technician + JobRole.xManufacturing.Director + 
                  JobRole.xResearch.Director + JobRole.xResearch.Scientist + 
                  JobRole.xSales.Executive + MaritalStatus.xSingle + 
                  StockOptionLevel.x1 + EnvironmentSatisfaction.xLow + EnvironmentSatisfaction.xVery.High + 
                  JobSatisfaction.xLow + JobSatisfaction.xVery.High + WorkLifeBalance.xBest + 
                  WorkLifeBalance.xBetter + WorkLifeBalance.xGood + WorkLifeBalance.xNo.response + 
                  JobInvolvement.xLow + JobInvolvement.xMedium + JobInvolvement.xVery.High, 
                family = "binomial", data = employee_final)
summary(model_10)
vif(model_10)

# Education.xCollege has non significant value, so is removed in model_11#
model_11 <- glm(formula = Attrition ~ Age + NumCompaniesWorked + 
                  TotalWorkingYears + TrainingTimesLastYear +  
                  YearsSinceLastPromotion + YearsWithCurrManager + BusinessTravel.xTravel_Frequently + 
                  Department.xResearch...Development + 
                  Department.xSales + Education.xDoctor +
                  EducationField.xOther +  
                  JobLevel.x5 + JobRole.xLaboratory.Technician + JobRole.xManufacturing.Director + 
                  JobRole.xResearch.Director + JobRole.xResearch.Scientist + 
                  JobRole.xSales.Executive + MaritalStatus.xSingle + 
                  StockOptionLevel.x1 + EnvironmentSatisfaction.xLow + EnvironmentSatisfaction.xVery.High + 
                  JobSatisfaction.xLow + JobSatisfaction.xVery.High + WorkLifeBalance.xBest + 
                  WorkLifeBalance.xBetter + WorkLifeBalance.xGood + WorkLifeBalance.xNo.response + 
                  JobInvolvement.xLow + JobInvolvement.xMedium + JobInvolvement.xVery.High, 
                family = "binomial", data = employee_final)
summary(model_11)
vif(model_11)

#EnvironmentSatisfaction.xVery.High has non significant value
model_12 <- glm(formula = Attrition ~ Age + NumCompaniesWorked + 
                  TotalWorkingYears + TrainingTimesLastYear +  
                  YearsSinceLastPromotion + YearsWithCurrManager + BusinessTravel.xTravel_Frequently + 
                  Department.xResearch...Development + 
                  Department.xSales + Education.xDoctor +
                  EducationField.xOther +  
                  JobLevel.x5 + JobRole.xLaboratory.Technician + JobRole.xManufacturing.Director + 
                  JobRole.xResearch.Director + JobRole.xResearch.Scientist + 
                  JobRole.xSales.Executive + MaritalStatus.xSingle + 
                  StockOptionLevel.x1 + EnvironmentSatisfaction.xLow + 
                  JobSatisfaction.xLow + JobSatisfaction.xVery.High + WorkLifeBalance.xBest + 
                  WorkLifeBalance.xBetter + WorkLifeBalance.xGood + WorkLifeBalance.xNo.response + 
                  JobInvolvement.xLow + JobInvolvement.xMedium + JobInvolvement.xVery.High, 
                family = "binomial", data = employee_final)
summary(model_12)
vif(model_12)

#JobInvolvement.xVery.High has non significant value
model_13 <- glm(formula = Attrition ~ Age + NumCompaniesWorked + 
                  TotalWorkingYears + TrainingTimesLastYear +  
                  YearsSinceLastPromotion + YearsWithCurrManager + BusinessTravel.xTravel_Frequently + 
                  Department.xResearch...Development + 
                  Department.xSales + Education.xDoctor +
                  EducationField.xOther +  
                  JobLevel.x5 + JobRole.xLaboratory.Technician + JobRole.xManufacturing.Director + 
                  JobRole.xResearch.Director + JobRole.xResearch.Scientist + 
                  JobRole.xSales.Executive + MaritalStatus.xSingle + 
                  StockOptionLevel.x1 + EnvironmentSatisfaction.xLow + 
                  JobSatisfaction.xLow + WorkLifeBalance.xBest + 
                  WorkLifeBalance.xBetter + WorkLifeBalance.xGood + WorkLifeBalance.xNo.response + 
                  JobInvolvement.xLow + JobInvolvement.xMedium + JobInvolvement.xVery.High, 
                family = "binomial", data = employee_final)
summary(model_13)
vif(model_13)

#EducationField.xOther
model_14 <- glm(formula = Attrition ~ Age + NumCompaniesWorked + 
                  TotalWorkingYears + TrainingTimesLastYear +  
                  YearsSinceLastPromotion + YearsWithCurrManager + BusinessTravel.xTravel_Frequently + 
                  Department.xResearch...Development + 
                  Department.xSales + Education.xDoctor +
                  JobLevel.x5 + JobRole.xLaboratory.Technician + JobRole.xManufacturing.Director + 
                  JobRole.xResearch.Director + JobRole.xResearch.Scientist + 
                  JobRole.xSales.Executive + MaritalStatus.xSingle + 
                  StockOptionLevel.x1 + EnvironmentSatisfaction.xLow + 
                  JobSatisfaction.xLow + WorkLifeBalance.xBest + 
                  WorkLifeBalance.xBetter + WorkLifeBalance.xGood + WorkLifeBalance.xNo.response + 
                  JobInvolvement.xLow + JobInvolvement.xMedium + JobInvolvement.xVery.High, 
                family = "binomial", data = employee_final)
summary(model_14)
vif(model_14)

#StockOptionLevel.x1
model_15 <- glm(formula = Attrition ~ Age + NumCompaniesWorked + 
                  TotalWorkingYears + TrainingTimesLastYear +  
                  YearsSinceLastPromotion + YearsWithCurrManager + BusinessTravel.xTravel_Frequently + 
                  Department.xResearch...Development + 
                  Department.xSales + Education.xDoctor +
                  JobLevel.x5 + JobRole.xLaboratory.Technician + JobRole.xManufacturing.Director + 
                  JobRole.xResearch.Director + JobRole.xResearch.Scientist + 
                  JobRole.xSales.Executive + MaritalStatus.xSingle + 
                  EnvironmentSatisfaction.xLow + 
                  JobSatisfaction.xLow + WorkLifeBalance.xBest + 
                  WorkLifeBalance.xBetter + WorkLifeBalance.xGood + WorkLifeBalance.xNo.response + 
                  JobInvolvement.xLow + JobInvolvement.xMedium + JobInvolvement.xVery.High, 
                family = "binomial", data = employee_final)
summary(model_15)
vif(model_15)

#final model#
final_model <- model_15 # since all the variables have significant p value, no more variables can be removed from the model, therefore, this forms the final model#

#----model evaluation---#
#predicted probability for Attrition for test data#

test_predict = predict(final_model, type = "response", newdata = test[,-2]) #predicting using test dataset with attrition column#
summary(test_predict)

test$prob <- test_predict # we get the probability of attrition of each employee#
View(test)

#Using probability cuttoff for attrition as 50%#
test_predict_attrition_at50 <- factor(ifelse(test_predict >= 0.50, "Yes", "No"))
test_actual_attrition <- factor(ifelse(test$Attrition == 1, "Yes", "No"))

table(test_actual_attrition, test_predict_attrition_at50)

#Determining Accuracy for cutoff of 50%#
test_conf <- confusionMatrix(test_predict_attrition_at50, test_actual_attrition, positive = "Yes")
test_conf

perform_fn <- function(cutoff) 
{
  predicted_attr <- factor(ifelse(test_predict >= cutoff, "Yes", "No"))
  conf <- confusionMatrix(predicted_attr, test_actual_attrition, positive = "Yes")
  acc <- conf$overall[1]
  sens <- conf$byClass[1]
  spec <- conf$byClass[2]
  out <- t(as.matrix(c(sens, spec, acc))) 
  colnames(out) <- c("sensitivity", "specificity", "accuracy")
  return(out)
}

#Creating cutoff values from 0.001493 to 0.901852 for plotting and initiallizing a matrix of 100 X 3.

# Summary of test probability

summary(test_predict)


s = seq(.01,.90,length=100)

OUT = matrix(0,100,3)
for(i in 1:100)
{
  OUT[i,] = perform_fn(s[i])
} 

plot(s, OUT[,1],xlab="Cutoff",ylab="Value",cex.lab=1.5,cex.axis=1.5,ylim=c(0,1),type="l",lwd=2,axes=FALSE,col=2)
axis(1,seq(0,1,length=5),seq(0,1,length=5),cex.lab=1.5)
axis(2,seq(0,1,length=5),seq(0,1,length=5),cex.lab=1.5)
lines(s,OUT[,2],col="darkgreen",lwd=2)
lines(s,OUT[,3],col=4,lwd=2)
box()
legend(0,.50,col=c(2,"darkgreen",4,"darkred"),lwd=c(2,2,2,2),c("Sensitivity","Specificity","Accuracy"))


cutoff <- s[which(abs(OUT[,1]-OUT[,2])<0.01)]
cutoff

# Let's choose a cutoff value of 0.169596 for final model
test_cutoff_attr <- factor(ifelse(test_predict >= 0.1696, "Yes", "No"))
conf_final <- confusionMatrix(test_cutoff_attr, test_actual_attrition, positive = "Yes")
acc <- conf_final$overall[1]

sens <- conf_final$byClass[1]

spec <- conf_final$byClass[2]

acc

sens

spec

View(test)

test_cutoff_attr <- ifelse(test_cutoff_attr=="Yes",1,0)
test_actual_attrition <- ifelse(test_actual_attrition=="Yes",1,0)

### KS -statistic - Test Data ######

#on testing  data
pred_object_test<- prediction(test_cutoff_attr, test_actual_attrition)

performance_measures_test<- performance(pred_object_test, "tpr", "fpr")

ks_table_test <- attr(performance_measures_test, "y.values")[[1]] - 
  (attr(performance_measures_test, "x.values")[[1]])

max(ks_table_test)

# Lift & Gain Chart 

# plotting the lift chart
lift <- function(labels , predicted_prob,groups=10) {
  
  if(is.factor(labels)) labels  <- as.integer(as.character(labels ))
  if(is.factor(predicted_prob)) predicted_prob <- as.integer(as.character(predicted_prob))
  helper = data.frame(cbind(labels , predicted_prob))
  helper[,"bucket"] = ntile(-helper[,"predicted_prob"], groups)
  gaintable = helper %>% group_by(bucket)  %>%
    summarise_at(vars(labels ), funs(total = n(),
                                     totalresp=sum(., na.rm = TRUE))) %>%
    
    mutate(Cumresp = cumsum(totalresp),
           Gain=Cumresp/sum(totalresp)*100,
           Cumlift=Gain/(bucket*(100/groups))) 
  return(gaintable)
}

Attr_decile = lift(test_actual_attrition, test_predict, groups = 10)
Attr_decile
