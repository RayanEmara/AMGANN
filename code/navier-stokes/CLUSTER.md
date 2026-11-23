# Run Navier–Stokes on the Polimi students cluster

The commands assume your project is in `~/AMG-ANN` on the cluster. An NVIDIA L4 runs the double-precision CG and BoomerAMG solves for both velocity and pressure. The job uses one GPU with one solver process. Finite-element assembly, explicit convection, diagnostics, CNN training and the two threshold predictions run on CPU. Both baseline and ANN simulations use the same GPU backend and solver tolerance.

## From your workstation

Each colleague needs an active Polimi students-cluster account, DMAT VPN access and the SSH key registered for their own account. The cluster username is `u` followed by the eight-digit *codice persona*. Your workstation username can be different.

Choose an installed VPN client. Both use the department portal `gp-dmat-saml.vpn.polimi.it` and university browser sign-in. For first-time installation, see the [Polimi VPN instructions](https://www.ict.polimi.it/network/vpn/?lang=en) or the [gpclient installation guide](https://github.com/yuezk/GlobalProtect-openconnect#installation).

| Installed client | Connect from the local repository root | End the session |
|---|---|---|
| Terminal (`gpclient`, macOS/Linux) | `bash code/navier-stokes/cluster/vpn.sh cli connect` | In another terminal, `bash code/navier-stokes/cluster/vpn.sh cli disconnect` |
| Palo Alto GlobalProtect | Open the application, enter the portal and click **Connect**. On macOS, `bash code/navier-stokes/cluster/vpn.sh paloalto connect` opens it. | Click **Disconnect** in the application. |

From the **local repository root**, use a Bash or Zsh terminal with OpenSSH and rsync installed. On Windows, you can connect GlobalProtect in Windows and run the transfer commands in WSL if its network can reach the VPN. Enter your own cluster username at the prompt. This uses the login node directly and works with an empty SSH configuration. If you registered a key with a non-default filename, load it into your SSH agent first with `ssh-add /path/to/your/private-key`.

```bash
printf 'Polimi cluster username (u followed by your codice persona): '
IFS= read -r cluster_user
cluster_login="${cluster_user}@10.78.18.100"
ssh "$cluster_login" 'mkdir -p ~/AMG-ANN'
rsync -avR --exclude='__pycache__/' --exclude='.pytest_cache/' --exclude='.DS_Store' \
  code/navier-stokes/ environment/navier-stokes-gpu.def "$cluster_login:AMG-ANN/"
ssh "$cluster_login"
```

If you already use a working SSH alias for your cluster account, you can set `cluster_login` to that alias instead. On first login, verify the server fingerprint through the cluster administrator. A public-key rejection calls for checking your username and registered key. An account-expiration message calls for contacting the administrator.

## Once logged in

Start each new login shell with:

```bash
. /etc/profile.d/pbs.sh
cd ~/AMG-ANN
qstat -Qf gpu
```

Submit from this directory. The login node handles transfers and scheduling. PBS runs the build and simulations on compute nodes.

**First-time build.** This image pins CUDA 12.4.1, PETSc/petsc4py 3.22.5 and builds hypre with CUDA for the L4 (SM 89). It includes CPU PyTorch. Building needs outbound access to the package and source registries and cluster support for Apptainer fakeroot. Check the queue's walltime/resource limits before submitting. Build time depends on downloads and the cluster.

```bash
build_id=$(qsub code/navier-stokes/cluster/build.pbs)
echo "$build_id"
qstat -f "$build_id"
```

**Quick smoke test.** For a new image, submit with the dependency below. PBS will start it only after the build exits successfully. On subsequent visits with an existing image, use `smoke_id=$(qsub code/navier-stokes/cluster/smoke.pbs)` instead.

```bash
smoke_id=$(qsub -W "depend=afterok:$build_id" code/navier-stokes/cluster/smoke.pbs)
echo "$smoke_id"
```

The smoke job checks the numerical solver against direct and CPU references, exercises both GPU operators, collects small timing curves, trains a three-epoch CNN and executes a fixed-vs-ANN comparison on a held-out 12×12 mesh. Its purpose is to catch installation and correctness problems. The job requests ten minutes.

**Full run.** Queue this while the smoke job is pending or running:

```bash
full_id=$(qsub -W "depend=afterok:$smoke_id" code/navier-stokes/cluster/full.pbs)
echo "$full_id"
qstat -u "$(whoami)"
```

If the smoke job already finished and you verified its output, submit directly with `qsub code/navier-stokes/cluster/full.pbs`. The full preset benchmarks five thresholds with three repetitions per matrix, uses 24 time steps, and trains for 150 epochs with validation checkpoint selection. Training meshes have 24, 40 and 64 cells per side, validation uses 80, and five paired comparisons use a held-out 96×96 mesh. The final stage checks manufactured-solution convergence. The two-hour walltime is an initial request, not a measured runtime.

## Outputs and troubleshooting

PBS writes joined logs to the submission directory (for example `ns_smoke.o1234`). The first lines identify the compute node and absolute result directory. Each job derives the work path from the account running it and copies results from local scratch when it exits. A hard scheduler kill can prevent the copy. For a completed experiment, inspect `gpu.csv`, `dataset/metadata.json`, `model.training.json`, and `comparison/comparison.json`. The report records the actual GPU matrix/vector types, selected thresholds, solution differences, and median full simulation times. Full runs also produce `convergence/convergence.json`.

To download a run, return to your local terminal with `exit`. Use the same `cluster_login` as for the upload (set it again if you opened a new shell). Paste the full path printed after `outputs:` in your PBS log:

```bash
printf 'Absolute cluster result directory: '
IFS= read -r run_dir
scp -r "${cluster_login:?Set cluster_login to your cluster SSH destination}:${run_dir:?Enter the result directory}" .
```

Keep a copy of important data outside the cluster. Once PBS accepts a batch job it can continue after SSH and the VPN disconnect.

If the build queue rejects a request, inspect `qstat -Qf gpu` and adjust the PBS resource directive to an allowed value. If the site disallows fakeroot or outbound builds, build the same definition on an authorized x86-64 Linux Apptainer builder with NVIDIA support and copy the resulting SIF to `/work/$USER/amg-ann/ns-l4.sif`. If the site supplies an equivalent image, submit either job with `qsub -v NS_SIF=/absolute/path/image.sif ...`; the job expects Python at `/opt/ns/bin/python`. Check that the node driver supports the image's CUDA toolkit. CUDA jobs fail on missing GPU support. A CPU-trained model belongs to a different timing backend and is rejected.

References: [PETSc BoomerAMG on GPUs](https://petsc.org/release/manualpages/PC/PCHYPRE/), [hypre GPU-supported options](https://hypre.readthedocs.io/en/latest/solvers-boomeramg.html), [Apptainer NVIDIA support](https://apptainer.org/docs/user/latest/gpu.html).
