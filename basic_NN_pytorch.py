{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "name": "Untitled6.ipynb",
      "provenance": [],
      "authorship_tag": "ABX9TyPjBsHv5uzcQCpF/XZO7bFY",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "accelerator": "GPU"
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/Raj-1999/Unsupervised_instrument_clustering/blob/master/basic%20NN%20with%20pytorch\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "MTEgE3EdHAl2",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "import torch\n",
        "import torch.nn.functional as F"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "mpdJmbLPHW5l",
        "colab_type": "code",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 34
        },
        "outputId": "1476ee76-b968-468c-92ee-1934ca2715f0"
      },
      "source": [
        "x=torch.randn(200,10)\n",
        "y1=torch.ones(100,1)\n",
        "y2=torch.zeros(100,1)\n",
        "y=torch.cat((y1,y2),0)\n",
        "epoch=450\n",
        "learning_rate=0.03\n",
        "y.shape"
      ],
      "execution_count": 3,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "torch.Size([200, 1])"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 3
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "rgjjIWKZHx-D",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "class Net(torch.nn.Module):\n",
        "  def __init__(self, n_input, n_hidden, n_output):\n",
        "    super(Net,self).__init__()\n",
        "    self.hidden=torch.nn.Linear(n_input,n_hidden)\n",
        "    self.output=torch.nn.Linear(n_hidden,n_output)\n",
        "  def forward(self, input):\n",
        "    hidden=F.relu(self.hidden(input))\n",
        "    output=self.output(hidden)\n",
        "    return output"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "TjIB3BfqO1RF",
        "colab_type": "code",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 86
        },
        "outputId": "7fd0a619-e62e-4faa-b92a-c9dd2ae2a858"
      },
      "source": [
        "net=Net(1,15,1)\n",
        "print(net)"
      ],
      "execution_count": 10,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "Net(\n",
            "  (hidden): Linear(in_features=1, out_features=15, bias=True)\n",
            "  (output): Linear(in_features=15, out_features=1, bias=True)\n",
            ")\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "b04c701No1ji",
        "colab_type": "code",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 120
        },
        "outputId": "94a5837f-e04e-4a01-bc3d-2861a23bd3d1"
      },
      "source": [
        "seq_net=torch.nn.Sequential(torch.nn.Linear(10,20),torch.nn.ReLU(),torch.nn.Linear(20,1),torch.nn.Sigmoid())\n",
        "print(seq_net)"
      ],
      "execution_count": 28,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "Sequential(\n",
            "  (0): Linear(in_features=10, out_features=20, bias=True)\n",
            "  (1): ReLU()\n",
            "  (2): Linear(in_features=20, out_features=1, bias=True)\n",
            "  (3): Sigmoid()\n",
            ")\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "g07ud3GJpqu7",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "loss_func=torch.nn.CrossEntropyLoss()\n",
        "Optimizer=torch.optim.SGD(seq_net.parameters(),lr=learning_rate,momentum=0.9)"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "4YsNkHtATPma",
        "colab_type": "code",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 397
        },
        "outputId": "d158a505-7301-4810-b49e-f82caa418cc2"
      },
      "source": [
        "\n",
        "\n",
        "for i in range(epoch):\n",
        "  output=seq_net(x) #prediction\n",
        "  loss=loss_func(output,y)\n",
        "  Optimizer.zero_grad()\n",
        "  loss.backward()\n",
        "\n",
        "  Optimizer.step()\n",
        "  if i%10==0:\n",
        "    print(i,\"th loss:\", loss.squeeze())"
      ],
      "execution_count": 30,
      "outputs": [
        {
          "output_type": "error",
          "ename": "RuntimeError",
          "evalue": "ignored",
          "traceback": [
            "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
            "\u001b[0;31mRuntimeError\u001b[0m                              Traceback (most recent call last)",
            "\u001b[0;32m<ipython-input-30-db4233f866bc>\u001b[0m in \u001b[0;36m<module>\u001b[0;34m()\u001b[0m\n\u001b[1;32m      3\u001b[0m \u001b[0;32mfor\u001b[0m \u001b[0mi\u001b[0m \u001b[0;32min\u001b[0m \u001b[0mrange\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mepoch\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      4\u001b[0m   \u001b[0moutput\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0mseq_net\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mx\u001b[0m\u001b[0;34m)\u001b[0m \u001b[0;31m#prediction\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m----> 5\u001b[0;31m   \u001b[0mloss\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0mloss_func\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0moutput\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0my\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m      6\u001b[0m   \u001b[0mOptimizer\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mzero_grad\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      7\u001b[0m   \u001b[0mloss\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mbackward\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.6/dist-packages/torch/nn/modules/module.py\u001b[0m in \u001b[0;36m__call__\u001b[0;34m(self, *input, **kwargs)\u001b[0m\n\u001b[1;32m    548\u001b[0m             \u001b[0mresult\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_slow_forward\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m*\u001b[0m\u001b[0minput\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m**\u001b[0m\u001b[0mkwargs\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    549\u001b[0m         \u001b[0;32melse\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 550\u001b[0;31m             \u001b[0mresult\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mforward\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m*\u001b[0m\u001b[0minput\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m**\u001b[0m\u001b[0mkwargs\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    551\u001b[0m         \u001b[0;32mfor\u001b[0m \u001b[0mhook\u001b[0m \u001b[0;32min\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_forward_hooks\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mvalues\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    552\u001b[0m             \u001b[0mhook_result\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mhook\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0minput\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mresult\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.6/dist-packages/torch/nn/modules/loss.py\u001b[0m in \u001b[0;36mforward\u001b[0;34m(self, input, target)\u001b[0m\n\u001b[1;32m    930\u001b[0m     \u001b[0;32mdef\u001b[0m \u001b[0mforward\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0minput\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mtarget\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    931\u001b[0m         return F.cross_entropy(input, target, weight=self.weight,\n\u001b[0;32m--> 932\u001b[0;31m                                ignore_index=self.ignore_index, reduction=self.reduction)\n\u001b[0m\u001b[1;32m    933\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    934\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.6/dist-packages/torch/nn/functional.py\u001b[0m in \u001b[0;36mcross_entropy\u001b[0;34m(input, target, weight, size_average, ignore_index, reduce, reduction)\u001b[0m\n\u001b[1;32m   2315\u001b[0m     \u001b[0;32mif\u001b[0m \u001b[0msize_average\u001b[0m \u001b[0;32mis\u001b[0m \u001b[0;32mnot\u001b[0m \u001b[0;32mNone\u001b[0m \u001b[0;32mor\u001b[0m \u001b[0mreduce\u001b[0m \u001b[0;32mis\u001b[0m \u001b[0;32mnot\u001b[0m \u001b[0;32mNone\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   2316\u001b[0m         \u001b[0mreduction\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0m_Reduction\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mlegacy_get_string\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0msize_average\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mreduce\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m-> 2317\u001b[0;31m     \u001b[0;32mreturn\u001b[0m \u001b[0mnll_loss\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mlog_softmax\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0minput\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;36m1\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mtarget\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mweight\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;32mNone\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mignore_index\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;32mNone\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mreduction\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m   2318\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   2319\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.6/dist-packages/torch/nn/functional.py\u001b[0m in \u001b[0;36mnll_loss\u001b[0;34m(input, target, weight, size_average, ignore_index, reduce, reduction)\u001b[0m\n\u001b[1;32m   2113\u001b[0m                          .format(input.size(0), target.size(0)))\n\u001b[1;32m   2114\u001b[0m     \u001b[0;32mif\u001b[0m \u001b[0mdim\u001b[0m \u001b[0;34m==\u001b[0m \u001b[0;36m2\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m-> 2115\u001b[0;31m         \u001b[0mret\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mtorch\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_C\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_nn\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mnll_loss\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0minput\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mtarget\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mweight\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0m_Reduction\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mget_enum\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mreduction\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mignore_index\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m   2116\u001b[0m     \u001b[0;32melif\u001b[0m \u001b[0mdim\u001b[0m \u001b[0;34m==\u001b[0m \u001b[0;36m4\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   2117\u001b[0m         \u001b[0mret\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mtorch\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_C\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_nn\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mnll_loss2d\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0minput\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mtarget\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mweight\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0m_Reduction\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mget_enum\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mreduction\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mignore_index\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;31mRuntimeError\u001b[0m: 1D target tensor expected, multi-target not supported"
          ]
        }
      ]
    }
  ]
}
