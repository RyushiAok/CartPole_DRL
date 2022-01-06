namespace CartPole.DuelingNetwork


open DiffSharp 
open DiffSharp.Model  
open DiffSharp.Compose    
            
type DuelingNetwork (observationSize: int, hiddenSize: int, actionCount: int) = 
    inherit Model()  
    let fc1     = Linear(observationSize,hiddenSize) 
    let fc2     = Linear(hiddenSize,hiddenSize)
    let fc3Adv  = Linear(hiddenSize,actionCount)
    let fc3V    = Linear(hiddenSize,1)

    do 
        base.add([ fc1; fc2; fc3Adv; fc3V])  

    override _.forward x =
        let h1  = x  --> fc1 --> dsharp.relu   
        let h2  = h1 --> fc2 --> dsharp.relu    
        let adv = h2 --> fc3Adv
        let v   = h2 --> fc3V |> dsharp.expand [-1;adv.shape.[1]]  
        v + adv - adv.mean(1,keepDim=true).expand[-1;adv.shape.[1] ]    
 
 