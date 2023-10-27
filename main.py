from src.model import SimpleRegression
import src.utils.data_processing as dp
import src.processing as prep

import torch
from tqdm import tqdm


def train(x_train_tensor, y_train_tensor):
    train_tensor = torch.utils.data.TensorDataset(x_train_tensor, y_train_tensor)
    batch_size = 64
    train_loader = torch.utils.data.DataLoader(train_tensor, batch_size=batch_size, shuffle=True)



    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(net.parameters(), lr=0.0001, weight_decay=0.0001)
    num_epochs = 500

    for epoch in range(num_epochs):

        for x, y in tqdm(train_loader):
            # transfer data to GPU
            x = x.to(device)
            y = y.to(device)

            # initialize grad
            optimizer.zero_grad()

            # predict in train data
            y_pred = net(x)
            y_pred = y_pred.squeeze()

            # calculating loss RMSE
            loss = torch.sqrt(criterion(torch.log1p(y_pred), torch.log1p(y)))

            # calculating grad
            loss.backward()

            # modify params
            optimizer.step()

        print(f'Epoch {epoch} loss: {loss.item():.5f}')
        torch.save(net.state_dict(), "model.pkl")
    return net


if __name__ == '__main__':


    # load data
    pp = prep.Processing()
    df_train = pp.train
    df_test = pp.test

    # remove target class from x
    X = df_train.drop('SalePrice', axis=1)
    Y = df_train['SalePrice'].to_numpy()

    # apply preprocessing pipeline
    pipeline = pp.getPipeline(X)

    X_train_prepped = pipeline.fit_transform(X)
    X_test_prepped = pipeline.transform(df_test)

    #transform to tensor
    x_train_tensor = torch.tensor(X_train_prepped).float()
    y_train_tensor = torch.tensor(Y).float()
    x_test_tensor = torch.tensor(X_test_prepped).float()


    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    n_input = X_train_prepped.shape[1]

    net = SimpleRegression.SimpleRegressionNet(n_input, 1, 1024, 4)
    net = net.to(device)

    x_train_tensor = x_train_tensor.to(device)
    y_train_tensor = y_train_tensor.to(device)
    x_test_tensor = x_test_tensor.to(device)

    #net = train(x_train_tensor, y_train_tensor)
    net.load_state_dict(torch.load("model.pkl"))

    predictions = []
    for i in range(50):
        p = net(x_test_tensor)
        predictions.append(p)
    prediction = torch.mean(torch.stack(predictions), dim=0).detach()

    dp.writeOutput(prediction.numpy(), "output.csv")

    print("END")





