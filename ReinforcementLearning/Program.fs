namespace CartPole 
open System.Runtime.InteropServices
open DiffSharp 
open DiffSharp.Backends.Torch  

module Main =  
    [<EntryPoint>]
    let main argv =  
        NativeLibrary.Load(@"C:\libtorch\lib\torch.dll") |> ignore
        dsharp.config(backend=Backend.Torch, device=Device.CPU) 
        CartPole.Gui.appRun argv  