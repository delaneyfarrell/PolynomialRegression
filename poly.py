# for polynomial regression using sci-kit in python

def poly(x,y,deg):
    # split data
    from sklearn.model_selection import train_test_split
    Xtr, Xte, Ytr, Yte = train_test_split(x,y,test_size=0.2,random_state=0)
    # linear regression
    from sklearn.linear_model import LinearRegression
    lin_reg = LinearRegression()
    lin_reg.fit(x,y)
    # polynomial regression
    from sklearn.preprocessing import PolynomialFeatures
    poly_reg = PolynomialFeatures(degree=deg)
    X_poly = poly_reg.fit_transform(Xtr)
    poly_reg.fit(X_poly,Ytr)
    lin_reg_2 = LinearRegression()
    lin_reg_2.fit(X_poly,Ytr)
    yhat = lin_reg_2.predict(poly_reg.fit_transform(Xte))
    return Yte, yhat
