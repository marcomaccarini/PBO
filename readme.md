# Fiware
```console
cd FIWARE
docker-compose up -d
```

Test connection going into   [localhost:1026/version](http://localhost:1026/version)

```console
python createEntities.py
python createSubscriptions.py
```
These two scripts must returns code 201.

# PBO 

```console
docker-compose build --no-cache
docker-compose up

```

## Testing the connection to PBO
Test connection going into   [http://localhost:5000/index](http://localhost:5000/index), the service will return the message "it works".

## Testing the optimization
To test the optimization you need the following elements.

### 1. fvarsX: variables to be optimized.
PBO needs information about the variable to be optimized, its type(discrete or continuous) and its domain.
```console
"fvarsX": [
        {
            "name": "a",
            "type": "continuous",
            "domain": [
                -200.0,
                200.0
            ]
        },{
            "name": "b",
            "type": "discrete",
            "domain": [
                -200.0,
                200.0,
                1
            ]
        }
    ],
```
In the case above, we want to optimize 2 variables, the first is named "a", it is a continuous variable and it can assume values between -200 and 200;
the second variable is named "b" and it is discrete, it can assume values between -200 and 200 with step of 1;

### 2. fvarsY: model used
This field must always be specified in this way. At the moment it is not used, but it was decided to include it for possible future implementations in which more than one evaluation criterion could be specified.
```console
    "fvarsY": [
        {
            "name": "out1",
            "type": "continuous",
            "domain": []
        }
    ]
```
### 3. acquisition_optimizer_type: the type of acquisition function.
```console
 "acquisition_optimizer_type": "lbfgsb"
```
### 4. x: the points already tested
PBO needs as input the initial points that are already tested.
```console
    "x": [
        [
            -10,1
        ],
        [
            44,1
        ],
        ...,
        [
            -16,1
        ]
    ]
```
Since the variable to be optimized are 2, every field of this array contains 2 elements.
### 5. b_in: the initial comparison
PBO needs as input the initial comparisons.
```console
 "b_in": [
        [
            0,
            1,
            1
        ],
        [
            2,
            1,
            1
        ],
        ...,
        [
            10,
            12,
            1
        ]
    ],
```
In the above example, the first array states that point at the position 0 of "x" array is better than the point at the position 1 of the same array, to sum up:
x[0] is better than x[1], x[2] is better than x[1], x[10] is better than x[12], 

### 5. b_best:  the best point seen so far
PBO needs as input the best point seen so far.
```console
"b_best": [
    10
]
```
In this case the best point seen so far is the point at the position 10 of "x" array, in other words the best point is x[10].


### 6. delta: exploration parameter
This parameter can assume value between 0 and 10. The higher this value is, the more PBO will explore.
```console
"delta": 3
```

### 7. the entire json file to post
It possible to see the entire json message in  [example.json](example.json)

