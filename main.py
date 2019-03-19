from body_AB import *

save_plots = False
debug      = False
n_epochs   = 5 + 1
y_var      = 'lt' 
EVALUATE   = True

torch.set_default_tensor_type('torch.cuda.FloatTensor')

#raw_data = pd.read_csv(os.path.join("data", "nasdaq100_padding.csv"), nrows=100 if debug else None)
raw_data = pd.read_csv(os.path.join("data", "prova.csv.zip"), nrows=200000 if debug else None)
logger.info(f"Shape of data: {raw_data.shape}.\nMissing in data: {raw_data.isnull().sum().sum()}.")
targ_cols  = (y_var,)
data, scaler = preprocess_data(raw_data, targ_cols)
da_rnn_kwargs = {"batch_size": 128, "T": 30}
config, model = da_rnn(data, n_targs=len(targ_cols), learning_rate=.001, **da_rnn_kwargs)
iter_loss, epoch_loss = train(model, data, config, n_epochs=n_epochs, save_plots=save_plots)

print("Finishing", end='\r')

final_y_pred = predict(model, data, config.train_size, config.batch_size, config.T)
print("Finishing.", end='\r')

plt.figure()
plt.semilogy(range(len(iter_loss)), iter_loss)
utils.save_or_show_plot("iter_loss.png", save_plots)

print("Finishing..", end='\r')

plt.figure()
plt.semilogy(range(len(epoch_loss)), epoch_loss)
utils.save_or_show_plot("epoch_loss.png", save_plots)

print("Finishing...", end='\r')

plt.figure()
plt.plot(final_y_pred, label='Predicted')
plt.plot(data.targs[config.train_size:], label="True")
plt.legend(loc='upper left')
utils.save_or_show_plot("final_predicted.png", save_plots)

print("Finishing....", end='\r')

with open(os.path.join("data", "da_rnn_kwargs.json"), "w") as fi:
    json.dump(da_rnn_kwargs, fi, indent=4)

joblib.dump(scaler, os.path.join("data", "scaler.pkl"))
torch.save(model.encoder.state_dict(), os.path.join("data", "encoder.torch"))
torch.save(model.decoder.state_dict(), os.path.join("data", "decoder.torch"))

np.save("pred.npy", final_y_pred)

print("Finishing....", end='\r')

if EVALUATE:
    from sklearn.metrics import mean_squared_error
    from sklearn.metrics import cohen_kappa_score
    from sklearn.metrics import f1_score
    from sklearn.metrics import precision_score
    from sklearn.metrics import recall_score

    df = raw_data[[y_var]].copy()
    df = df.iloc[-len(final_y_pred):].copy()

    try:
        df['pred'] = final_y_pred
        s1 = precision_score(df[y_var], df['pred'], average='weighted') * 100
        s2 = recall_score   (df[y_var], df['pred'], average='weighted') * 100
        s3 = f1_score       (df[y_var], df['pred'], average='weighted') * 100
    except:
        df['prob'] = final_y_pred
        df['pred'] = np.where( df['prob']>1/3, 1, np.where( df['prob']<-1/3 ,-1,0)  )
        s1 = precision_score(df[y_var], df['pred'], average='weighted') * 100
        s2 = recall_score   (df[y_var], df['pred'], average='weighted') * 100
        s3 = f1_score       (df[y_var], df['pred'], average='weighted') * 100

    print("precision_score: {0:0.2f}%, \nrecall_score: {1:0.2f}%, \nf1_score: {2:0.2f}%".format(s1,s2,s3))