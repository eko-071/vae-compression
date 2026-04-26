# VAE Compression

Exploring image compression using Variational Autoencoders (VAEs) and benchmarking against classical codecs like JPEG and zlib.

## Project Structure

```
.
├── codec/ # the compression and decompression pipeline
├── evaluation/ # benchmarking scripts
├── models/ # one file per model version
├── notebooks/ # results exploration and documentations
├── README.md
├── requirements.txt
├── results/ # figures and logs
├── tests/ # tests to see if stuff works
└── training/ # figures and logs
```

## Dataset

MNIST to start. Probably CIFAR-10 as well for a more varied dataset, and then one more domain-specific dataset.