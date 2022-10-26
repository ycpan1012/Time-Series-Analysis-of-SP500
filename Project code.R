#######################################################
# ARIMA(p, d, q) model of Daily S&P 500 stock returns #
#######################################################
### preparation
# import library and data
library(astsa)
library(tseries)
library(forecast)  
load("Stocks.RData")

# define function for AIC -- from rugarch package definition
aic <- function(logliklihood, k, n) {
  result <- (-2 * logliklihood)/n + 2*k/n
  print(result)
  }
  
# define function for BIC - from rugarch package definition
bic <- function(logliklihood, k, n) {
  result <- (-2 * logliklihood)/n + k*log(n)/n
  print(result)
  }


### EDA
# initial plot
date <- time(Stocks)
target <- Stocks[,6]
plot(x=date, y=target, type='l',
     xlab='Date', ylab='Price',  main="Stock price of S&P500") 

# transformation
x1 <- diff(log(target))
plot(x=date, y=x1, type='l',
     xlab='Date', ylab='Return',  main="Stock return of S&P500") 

# ACF
acf2(x1, max.lag = 100)


### Split
training <- window(x1, start=c(2017, 2), end=c(2019, 56))
test <- window(x1, start=c(2019, 57), end=c(2019,66))


### build ARIMA(8,1,0)
# using Arima
arima1 <- Arima(training, order=c(8,0,0), method='ML')
arima1
arima1_aic <- aic(arima1$loglik,length(arima1$coef),length(training))
arima1_bic <- bic(arima1$loglik,length(arima1$coef),length(training))

# using sarima - diagnostic plot
sarima1 <- sarima(training,p=8, d=0, q=0)


### build ARIMA(0,1,8)
# using Arima 
arima2 <- Arima(training, order=c(0,0,8), method='ML',include.constant = TRUE)
arima2

# information criteria
arima2_aic <- aic(arima2$loglik,length(arima2$coef),length(training))
arima2_bic <- bic(arima2$loglik,length(arima2$coef),length(training))

# using sarima - diagnostic plot
sarima2 <- sarima(training, p=0, d=0, q=8)


### forecast
# using ARIMA(8,1,0)
arima_forecast <- forecast(arima1, h = 10)
plot(arima_forecast, xlim=c(2018.9,2019.3))
abline(h=0, lty=2, col='grey')

# another plot
arima1 %>%
  forecast(h=10) %>%
  autoplot(xlim=c(2018.9,2019.3)) + autolayer(test)

# take a look at the price data (with log)
training_logp <- subset(log(target), end=length(target)-10)
test_logp <- subset(log(target), start=length(target)-9)
logprice <- Arima(training_logp, c(0,1,8), include.constant = TRUE)
logprice %>%
  forecast(h=10) %>%
  autoplot(xlim=c(2018.9,2019.3)) + autolayer(test_logp)



#######################################################
# AR(1) + GARCH(1,1) model of Daily S&P 500 stock returns #
#######################################################
### preparation
#import library
library(xts)
library(rugarch)
library(MASS) 
library(EnvStats)
library(tsbox)
library(fGarch)

# convert ts to xts
x1_xts <- ts_xts(x1)

### fit an AR(1) + GARCH(1,1) w/ Gaussian 
# using fgarch
garch_fit <- garchFit(~ arma(1,0)+garch(1,1), data =diff(log(target)),cond.dist ="norm")
summary(garch_fit)

# disgnostic plot usiing ugarch
# specification
spec <- ugarchspec(
  mean.model = list(armaOrder = c(1, 0), include.mean = TRUE),
  # "sGARCH" is the usual GARCH model we defined:
  variance.model = list(model = "sGARCH", garchOrder = c(1, 1)),
  # We assume a normal model for e_t: 
  distribution.model="norm"
)

# fit
ar_garch <- ugarchfit(spec=spec, data = x1_xts, out.sample = 10)

# information criteria
ar_garch_aic <- aic(likelihood(ar_garch), length(ar_garch@fit$coef), length(x1)-10)
ar_garch_bic <- bic(likelihood(ar_garch), length(ar_garch@fit$coef), length(x1)-10)

# plot
plot(ar_garch, which=1) # 1: Series with 2 Conditional SD Superimposed
plot(ar_garch, which=8) # 8: Empirical Density of Standardized Residuals
plot(ar_garch, which=9) # 9: QQ-Plot of Standardized Residuals
plot(ar_garch, which=10) # 10: ACF of Standardized Residuals
plot(ar_garch, which=11) # 11: ACF of Squared Standardized Residuals

### fit an AR(1) + GARCH(1,1) w/ t-distribution
# fit residuals to t-distn
ehat <- as.numeric(residuals(ar_garch, standardize=TRUE))
fitdistr(x=ehat, densfun='t') 
qqPlot(x=ehat, distribution='t', 
       param.list=list(df=4), add.line=TRUE)

# using fgarch
garch_fit_t <- garchFit(~ arma(1,0)+garch(1,1), data =diff(log(target)),cond.dist ="std")
summary(garch_fit_t)

# diagnostic plots - using ugarch
# specification
spec_t <- ugarchspec(
  mean.model = list(armaOrder = c(1, 0), include.mean = TRUE),
  variance.model = list(model = "sGARCH", garchOrder = c(1, 1)),
  distribution.model="std"
)

# fit
ar_garch_t <- ugarchfit(spec=spec_t, data = x1_xts, out.sample = 10)

# information criteria
ar_garch_t_aic <- aic(likelihood(ar_garch_t), length(ar_garch_t@fit$coef), length(x1)-10)
ar_garch_t_bic <- bic(likelihood(ar_garch_t), length(ar_garch_t@fit$coef), length(x1)-10)

# plot
plot(ar_garch_t, which=1)
plot(ar_garch_t, which=8)
plot(ar_garch_t, which=9)
plot(ar_garch_t, which=10)
plot(ar_garch_t, which=11)

### Forecasting

# There are two types of forecasts: 
# - The usual forecast in which we forecast the 
# next h values of our time series all at once based 
# on the information available to us now. The n.ahead
# argument below pertains to this kind of forecast, and
# we set n.ahead = h. 
fc1 <- ugarchforecast(fitORspec=ar_garch_t, n.ahead = 10)
plot(fc1, which=1)

# - In a rolling forecast, we make a 1-step-ahead
# forecast, then observe the next value in the series, 
# then we make another 1-step ahead forecast taking this
# new information account, and so on. The n.roll argument
# below determines how many times we do this. 
fc2 <- ugarchforecast(fitORspec=ar_garch_t, n.ahead = 10, n.roll=10)
plot(fc2, which=1)
plot(fc2, which=2)
plot(fc2, which=3)
plot(fc2, which=4)
