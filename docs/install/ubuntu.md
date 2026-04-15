# Installing on Ubuntu (22.04 / 24.04)

The [`scripts/setup/ubuntu.sh`](../../scripts/setup/ubuntu.sh) helper does
everything below in one step:

```bash
bash scripts/setup/ubuntu.sh               # CPU-only
ENABLE_CUDA=1 bash scripts/setup/ubuntu.sh # + CUDA toolkit
ENABLE_SYCL=1 bash scripts/setup/ubuntu.sh # + Intel oneAPI
INSTALL_LINTERS=1 bash scripts/setup/ubuntu.sh # + clang-tidy/cppcheck/iwyu
```

## Manual install

```bash
sudo apt-get update
sudo apt-get install -y \
    build-essential meson ninja-build pkg-config nasm \
    python3-venv python3-pip \
    clang clang-format clang-tidy cppcheck doxygen
```

### CUDA (optional)

Requires an NVIDIA GPU. Use the official NVIDIA repo — Ubuntu's `nvidia-cuda-toolkit` is often outdated:

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get install -y cuda-toolkit-12-6
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
```

### SYCL / Intel oneAPI (optional)

```bash
wget -qO- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB \
    | sudo gpg --dearmor -o /usr/share/keyrings/oneapi-archive-keyring.gpg
echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" \
    | sudo tee /etc/apt/sources.list.d/oneAPI.list
sudo apt-get update
sudo apt-get install -y intel-basekit
source /opt/intel/oneapi/setvars.sh
```

## Build

```bash
cd libvmaf
meson setup ../build \
    -Denable_cuda=true \
    -Denable_sycl=true
ninja -C ../build
```

Binary lands at `build/tools/vmaf`.

## Run the Netflix golden tests

```bash
make test-netflix-golden
```
