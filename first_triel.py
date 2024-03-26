import mlflow
def cal_sum(x,y):
    return x+y

if __name__=="__main__":
    print("Its running...")
    #starting the server of mlflow
    with mlflow.start_run():
        x,y=10,20
        z=cal_sum(x,y)

        #tracking the experiment with mlflow

        mlflow.log_param("x",x)
        mlflow.log_param("y",y)
        mlflow.log_metric("z",z)
        
    

    

