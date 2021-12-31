namespace CartPole
 
open DiffSharp.Util 
 
type Action = | Left | Right | Nothing

type [<Measure>] kg 
type [<Measure>] m 
type [<Measure>] s
type [<Measure>] N = kg m / s^2 

type Apparatus = { 
    x         : float<m>
    dx        : float<m/s>
    theta     : float
    dtheta    : float</s>
    massCart  : float<kg>
    massPole  : float<kg>
    length    : float<m>
    g         : float<m/s^2>
    tau       : float<s>
    forceMag  : float<N>  
    width     : float
    height    : float   
    scale     : float
    elappesed : int 

} with 
    member this.Move action  = 
        // http://www.rel.hiroshima-u.ac.jp/inverted/index.html
        // https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py
        let force =
            match action with
            | Left  ->  -this.forceMag
            | Right -> this.forceMag
            | Nothing -> 0.0<N>
        let sinTheta = this.theta |> float |> sin 
        let cosTheta = this.theta |> float |> cos    
        let poleMassLength =  this.massPole * this.length 
        let massTotal = this.massPole + this.massCart
        let tmp = (force + poleMassLength * sinTheta * this.dtheta * this.dtheta) / massTotal
        let ddtheta =
            (this.g * sinTheta - cosTheta * tmp) 
            / (this.length * (4.0 / 3.0 - this.massPole * cosTheta * cosTheta / massTotal)) 
        let ddx = tmp - poleMassLength * ddtheta * cosTheta / massTotal

        { this with
            x         = this.x + this.dx * this.tau
            dx        = this.dx + ddx * this.tau
            theta     = this.theta + this.dtheta * this.tau
            dtheta    = this.dtheta + ddtheta * this.tau  
            elappesed = this.elappesed + 1  
        } 

    member this.Observations()   =   
        [|  float this.x
            float this.dx
            float this.theta
            float this.dtheta
        |] 
    
    member this.Failed ()  = 
        this.theta < -0.21 
        || 0.21 < this.theta 
        || float this.x < -2.4
        || 2.4 < float this.x   


module Apparatus =
    let init () =
        {   x         = Random.Double() * 0.1<m> - 0.05<m>
            dx        = Random.Double() * 0.1<m/s> - 0.05<m/s>
            theta     = Random.Double() * 0.1 - 0.05
            dtheta    = Random.Double() * 0.1</s>  - 0.05</s> 
            massCart  = 1.0<kg>
            massPole  = 0.1<kg>
            length    = 0.5<m>  
            g         = 9.8<m/s^2>
            tau       = 0.02<s>
            forceMag  = 10.0<N>  
            elappesed = 0 
            width     = 4.8 * 120.0
            height    = 300.0   
            scale     = 120.0
        }  

open FSharp.Control.Reactive   

[<AbstractClass>]
type Environment() = 
    abstract member Steps : int
    abstract member Observations : unit -> float[]
    abstract member Reflect : Action -> float[] * float * bool
    abstract member IsDone : unit -> bool
    abstract member Elappsed : unit -> int
    abstract member Reset : unit -> unit

type Env (?steps:int) = 
    inherit Environment()

    let steps = defaultArg steps 200 

    let subject = Subject.behavior <| Apparatus.init  ()  

    member _.Subject = subject

    override _.Steps = steps

    override _.Observations() = subject.Value.Observations()
    
    override _.Elappsed() = subject.Value.elappesed

    override _.Reset() = 
        subject.OnNext <| Apparatus.init ()
    
    member this.Failed () = // dueling ？
        subject.Value.Failed()  

    override this.IsDone() = 
        subject.Value.Failed() || this.Elappsed() >= steps

    override this.Reflect(action:Action)  =  

        subject.OnNext <| subject.Value.Move action 

        let reward =  
            if this.IsDone() then  
                if subject.Value.Failed() then -1.0 else 1.0 
            else 
                0.0    

        let isDone = this.IsDone() 

        subject.Value.Observations(), reward, isDone



open System.Net
open System.Net.Sockets
open System.Text
open System.Text.RegularExpressions
open System.Threading
open System.Reactive.Subjects 
open FSharp.Control 
open FSharp.Control.Reactive


type GymEnvironment(client:Socket, ?steps:int)=
    inherit Environment() 
    let steps = defaultArg steps 200  
    let regexObs = Regex("^o:(-?\d+\.\d+),(-?\d+\.\d+),(-?\d+\.\d+),(-?\d+\.\d+)$", RegexOptions.Compiled) 
    let regexAct = Regex("^r:(-?\d+\.\d+),(-?\d+\.\d+),(-?\d+\.\d+),(-?\d+\.\d+),(-?\d+\.\d+),(-?\d+\.\d+)$", RegexOptions.Compiled)   
    let (|Observations|ActionResult|None|)  (input:string) =

        if input.Length = 0 then None else  
        match input[0] with
        | 'o' -> Observations (
            regexObs.Match(input).Groups 
            |> Seq.toArrayQuick |> fun ary -> ary[1..] 
            |> Array.map(fun t -> t.ToString() |> System.Double.Parse) )  
        | 'r' -> ActionResult (
            regexAct.Match(input) .Groups 
            |> Seq.toArrayQuick |> fun ary -> ary[1..] 
            |> Array.map(fun t -> t.ToString() |> System.Double.Parse)
            |> fun ary -> ary[0..3], ary[4..]
            )  
        | _ -> None  
    
    let mutable elappsed = 0
    let mutable recievedAR = false
    let observations = Subject.behavior <| Array.create 4 0.0   
    let actionResult = Subject.behavior <| (0., false)

    do   
        async {
            let buffer = Array.create 128 (byte 0)  
            let rec read() =
                let len = client.Receive(buffer) 
                Encoding.Default.GetString(buffer, 0, len) 
                |> function 
                    | Observations s -> observations.OnNext s
                    | ActionResult (obs, res) ->
                        observations.OnNext obs
                        let failed = 
                            let obs = observations.Value
                            let x,theta = obs[0],obs[2] 
                            theta < -0.21 
                            || 0.21 < theta 
                            || float x < -2.4
                            || 2.4 < float x 
                        let isdone = failed || elappsed >= steps 
                        let reward =  
                            if isdone then  
                                if failed then -1.0 else 1.0 
                            else 
                                0.0    
                        let isDone = isdone
                        actionResult.OnNext (reward, isDone)
                        //reward <- -s[0]; 
                        //isDone <- s[1] = 1.0; 
                        recievedAR <- true
                    | None -> () 
                 

                read()
            read()
        }
        |> Async.Start

    
    override _.Steps = steps

    override _.Observations() = 
    
        observations.Value 
    
    override _.Elappsed() = elappsed

    override _.Reset() = 
        elappsed <- 0
        //isDone <- false
        "reset" |> Encoding.Default.GetBytes |> client.Send |> ignore 
        Thread.Sleep 50
    
    member this.Failed () = 
        let obs = observations.Value
        let x,theta = obs[0],obs[2] 
        theta < -0.21 
        || 0.21 < theta 
        || float x < -2.4
        || 2.4 < float x    

    override this.IsDone() = 
        // isDone
        this.Failed() || this.Elappsed() >= steps

    override this.Reflect(action:Action)  =  
        elappsed <- elappsed + 1  
        recievedAR <- false
        do
            match action with 
            | Left -> "0"
            | Right -> "1"
            | Nothing -> failwithf "this command is invalid in Gym"
            |> Encoding.Default.GetBytes |> client.Send |> ignore 
            //Thread.Sleep 50
        
        let reward, isDone =  
            while recievedAR = false do
                Thread.Sleep 1// ActionResultの受け取りを待機 
            actionResult.Value
        observations.Value,  reward, isDone 
         

type ClientCommand = 
    | Reset
    | Action of string 

type ClientMessage = 
    | Observations of float[] 
    | ActionResult of float[] * float * bool

type GymEnvironment'(message: AsyncSeq<ClientMessage>, command:BehaviorSubject<ClientCommand> , ?steps:int)=
    inherit Environment() 
    let steps = defaultArg steps 200  
    //let regexObs = Regex("^o:(-?\d+\.\d+),(-?\d+\.\d+),(-?\d+\.\d+),(-?\d+\.\d+)$", RegexOptions.Compiled) 
    //let regexAct = Regex("^r:(-?\d+\.\d+),(-?\d+\.\d+),(-?\d+\.\d+),(-?\d+\.\d+),(-?\d+\.\d+),(-?\d+\.\d+)$", RegexOptions.Compiled)   
    //let (|Observations|ActionResult|None|)  (input:string) =

    //    if input.Length = 0 then None else  
    //    match input[0] with
    //    | 'o' -> Observations (
    //        regexObs.Match(input).Groups 
    //        |> Seq.toArrayQuick |> fun ary -> ary[1..] 
    //        |> Array.map(fun t -> t.ToString() |> System.Double.Parse) )  
    //    | 'r' -> ActionResult (
    //        regexAct.Match(input) .Groups 
    //        |> Seq.toArrayQuick |> fun ary -> ary[1..] 
    //        |> Array.map(fun t -> t.ToString() |> System.Double.Parse)
    //        |> fun ary -> ary[0..3], ary[4..]
    //        )  
    //    | _ -> None  
    
    let mutable elappsed = 0
    let mutable recievedAR = false
    let observations = Subject.behavior <| Array.create 4 0.0   
    let actionResult = Subject.behavior <| (0., false)

    do   
        //printfn "AAAAA"
        message
        |> AsyncSeq.iter(fun msg -> 
            printfn "msg: %A" msg
            match msg with 
            | ClientMessage.Observations obs -> observations.OnNext obs
            | ClientMessage.ActionResult (obs, reward, isDone) -> 
                observations.OnNext obs
                let failed = 
                    let obs = observations.Value
                    let x,theta = obs[0],obs[2] 
                    theta < -0.21 
                    || 0.21 < theta 
                    || float x < -2.4
                    || 2.4 < float x 
                let isdone = failed || elappsed >= steps 
                let reward =  
                    if isdone then  
                        if failed then -1.0 else 1.0 
                    else 
                        0.0    
                let isDone = isdone
                actionResult.OnNext (reward, isDone) 
                recievedAR <- true
        )
        |> Async.Start
        |> ignore

        //async {
        //    let buffer = Array.create 128 (byte 0)  
        //    let rec read() =
        //        let len = command.Receive(buffer) 
        //        Encoding.Default.GetString(buffer, 0, len) 
        //        |> function 
        //            | Observations s -> observations.OnNext s
        //            | ActionResult (obs, res) ->
        //                observations.OnNext obs
        //                let failed = 
        //                    let obs = observations.Value
        //                    let x,theta = obs[0],obs[2] 
        //                    theta < -0.21 
        //                    || 0.21 < theta 
        //                    || float x < -2.4
        //                    || 2.4 < float x 
        //                let isdone = failed || elappsed >= steps 
        //                let reward =  
        //                    if isdone then  
        //                        if failed then -1.0 else 1.0 
        //                    else 
        //                        0.0    
        //                let isDone = isdone
        //                actionResult.OnNext (reward, isDone)
        //                //reward <- -s[0]; 
        //                //isDone <- s[1] = 1.0; 
        //                recievedAR <- true
        //            | None -> ()  
        //        read()
        //    read()
        //}
        //|> Async.Start

    
    override _.Steps = steps

    override _.Observations() = 
    
        observations.Value 
    
    override _.Elappsed() = elappsed

    override _.Reset() = 
        elappsed <- 0
        //isDone <- false
        //"reset" |> Encoding.Default.GetBytes |> client.Send |> ignore 
        command.OnNext ClientCommand.Reset
        Thread.Sleep 50
    
    member this.Failed () = 
        let obs = observations.Value
        let x,theta = obs[0],obs[2] 
        theta < -0.21 
        || 0.21 < theta 
        || float x < -2.4
        || 2.4 < float x    

    override this.IsDone() = 
        // isDone
        this.Failed() || this.Elappsed() >= steps

    override this.Reflect(action:Action)  =  
        elappsed <- elappsed + 1  
        recievedAR <- false
        do
            match action with 
            | Left -> "0"
            | Right -> "1"
            | Nothing -> failwithf "this command is invalid in Gym"
            //|> Encoding.Default.GetBytes // |> client.Send |> ignore 
            |> ClientCommand.Action
            |> command.OnNext
        
        let reward, isDone =  
            while recievedAR = false do
                Thread.Sleep 1// ActionResultの受け取りを待機 
            actionResult.Value
        observations.Value,  reward, isDone 
         

 