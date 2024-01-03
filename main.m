% Initialization
map = imread('random_map.bmp');
map = im2bw(map);
%map = zeros(500, 500);
%map = ones(500, 500);
start = [1, 1]; % Define start point
finish = [size(map, 1), size(map, 2)]; % Define finish point

% Genetic Algo Params
turningPoints = 8;
crossoverRate = 0.55;
mutationRate = 0.66;

populationSize = 470;
numGenerations = 100;

%populationSize = round(size(map, 1) * 2 / turningPoints); % More turning points -> Smaller Population
%numGenerations = round(size(map, 1) * size(map, 2) * 0.2 * (1 - crossoverRate)); % More CR -> Less Generations

% Ask user mode
while true
  SelectionChoice = input('Selection Mode: 1.Roulette_Wheel 2.Tournament 3.Rank_Based');
  
  % Validate input
  if ~isnumeric(SelectionChoice) || SelectionChoice < 1 || SelectionChoice > 3
    disp('Invalid input');
    continue;
  else
    break;
  end
end
while true
  CrossOverChoice = input('CrossOver Mode: 1.2K-point 2.Cycle');
  
  % Validate input
  if ~isnumeric(CrossOverChoice) || CrossOverChoice < 1 || CrossOverChoice > 2
    disp('Invalid input');
    continue;
  else
    break;
  end
end
while true
  MutationChoice = input('Mutation Mode: 1.Gaussian 2.Scramble');
  
  % Validate input
  if ~isnumeric(MutationChoice) || MutationChoice < 1 || MutationChoice > 2
    disp('Invalid input');
    continue;
  else
    break;
  end
end
fprintf("Running Config: %d, %d, %d - ", SelectionChoice, CrossOverChoice, MutationChoice)
%SelectionChoice = 3;
%CrossOverChoice = 2;
%MutationChoice = 2;
%meanFitnessPerGeneration = double(numGenerations);
%% 
% 1.Initialize Population

population = initializePopulationC(populationSize, start, finish, size(map), turningPoints, 85, 15);
% 2.Evaluate Initial Population
fitness = zeros(populationSize, 1);
fitness = objectiveFunction(population, map);
%displayResults(population, fitness, map, start, finish);
%%
% 3.GA Loop
tic;
for gen = 1:numGenerations
    % A. Selection
    parents = selection(population, fitness, SelectionChoice);
    % B. Crossoverx
    offspring = crossover(parents, crossoverRate, CrossOverChoice);
    % C. Mutation
    offspring = mutation(offspring, mutationRate, MutationChoice);
    % D. Accepting
    population = updateGA(offspring, populationSize, start, finish, size(map), turningPoints);
    % E. Evaluate Fitness of New Generation
    fitness = objectiveFunction(population, map);
    meanFitnessPerGeneration(gen) = mean(fitness);
    %disp(['Generation ', num2str(gen), ' - Mean Fitness: ', num2str(meanFitnessPerGeneration(gen))]);
end
toc;

% Plotting the mean fitness over generations
%figure;
%plot(meanFitnessPerGeneration);
%title('Mean Fitness over Generations');
%xlabel('Generation');
%ylabel('Mean Fitness');
% End Result
displayResults(population, fitness, map, start, finish);
[~, bestIdx] = max(fitness);
bestSolution = population(bestIdx, :, :);
plotRequiredPathA(map, bestSolution, start, finish);
%% 
% 
%% ------------------ Spawn Population Block ------------------

function population = initializePopulationA(populationSize, start, finish, mapSize, turningPoints)
    % Number of dimensions for each individual
    individualSize = [turningPoints, size(start,2)];
    population = zeros([populationSize, individualSize]);
    
    % Calculate the spacing and maximum offset so points are sequential
    spacing = floor(mapSize ./ (turningPoints + 1));
    maxOffset = floor(spacing / 2) + 1;

    % Generate each individual in the population
    for i = 1:populationSize
        individual = zeros(individualSize);

        for j = 1:turningPoints
            % Calculate base coordinates (evenly spaced)
            baseX = j * spacing(1);
            baseY = j * spacing(2);
            
            % Apply a random offset to each coordinate
            offsetX = baseX + randi([-1 1]) * randi(maxOffset);
            offsetY = baseY + randi([-1 1]) * randi(maxOffset);

            % Ensure the points are within the map boundaries
            offsetX = max(min(offsetX, mapSize(1)), 1);
            offsetY = max(min(offsetY, mapSize(2)), 1);

            % Check for obstacles and adjust if necessary
            while ~isPointValid(offsetX, offsetY, mapSize)
                offsetX = baseX + randi([-1 1]) * randi(maxOffset);
                offsetY = baseY + randi([-1 1]) * randi(maxOffset);
                offsetX = max(min(offsetX, mapSize(1)), 1);
                offsetY = max(min(offsetY, mapSize(2)), 1);
            end

            % Add the point to the individual
            individual(j, :) = [offsetX, offsetY];
        end

        % Add the individual to the population
        population(i, :, :) = individual;
    end
end
function populationNew = initializePopulationB(populationSize, start, finish, map, turningPoints)
    % Random Points on white space
    populationNew = zeros(populationSize, turningPoints, 2);
    populationNew(:, :, 1) = randi([start(1), finish(1)], populationSize, turningPoints);
    populationNew(:, :, 2) = randi([start(2), finish(2)], populationSize, turningPoints);
    populationNew = validateAndRegeneratePoints(populationNew, map, start, finish);
end
function population = initializePopulationC(populationSize, start, finish, mapSize, turningPoints, x, y)
    % Validate the weights
    if x < 0 || y < 0
        error('Weights x and y must be non-negative.');
    end

    % Normalize weights
    totalWeight = x + y;
    weightA = x / totalWeight;
    weightB = y / totalWeight;

    % Determine the number of rows from each population
    numFromA = round(populationSize * weightA);
    numFromB = populationSize - numFromA;  % Ensure total size remains as populationSize

    % Generate populations using initializePopulationA and initializePopulationB
    populationA = initializePopulationA(populationSize, start, finish, mapSize, turningPoints);
    populationB = initializePopulationB(populationSize, start, finish, mapSize, turningPoints);

    % Select the required number of rows from each population
    selectedA = populationA(1:numFromA, :, :);
    selectedB = populationB(1:numFromB, :, :);

    % Combine the selected rows to form the new population
    population = cat(1, selectedA, selectedB);  % Concatenate along the first dimension
end

function population = validateAndRegeneratePoints(population, map, start, finish)
    [populationSize, turningPoints, ~] = size(population);

    for i = 1:populationSize
        for j = 1:turningPoints
            % Ensure indices are within map bounds
            x = min(max(population(i, j, 1), 1), size(map, 2));
            y = min(max(population(i, j, 2), 1), size(map, 1));

            % Define the region based on the point's position in the sequence
            if j <= 3
                xRegion = [1, size(map, 2)/2]; % First 3 points near start
                yRegion = [1, size(map, 1)/2];
            elseif j > turningPoints - 3
                xRegion = [size(map, 2)/2, size(map, 2)]; % Last 3 points near finish
                yRegion = [size(map, 1)/2, size(map, 1)];
            else
                xRegion = [start(1), finish(1)]; % Middle points
                yRegion = [start(2), finish(2)];
            end

            % Adjust region to avoid exceeding map bounds
            xRegion = [max(xRegion(1), 1), min(xRegion(2), size(map, 2))];
            yRegion = [max(yRegion(1), 1), min(yRegion(2), size(map, 1))];

            % Regenerate point if it's invalid and within the defined region
            while map(y, x) == 0
                x = randi(xRegion);
                y = randi(yRegion);
                % Ensure indices are within map bounds
                x = min(max(x, 1), size(map, 2));
                y = min(max(y, 1), size(map, 1));
                population(i, j, :) = [x, y]; % Update the population
            end
        end
    end
end
function isValid = isPointValid(x, y, mapSize)
    % Check if the point (x, y) is within the map boundaries
    isValid = x >= 1 && x <= mapSize(1) && y >= 1 && y <= mapSize(2);
end
function updatedPopulation = updateGA(newPopulation, populationSize, start, finish, mapsize, turningPoints)
    if size(newPopulation, 1) < populationSize
        additionalPopulation = initializePopulationC(populationSize - size(newPopulation, 1), start, finish, mapsize, turningPoints, 15, 90);
        updatedPopulation = cat(1, newPopulation, additionalPopulation);
    else
        updatedPopulation = newPopulation;
    end
end
%% ------------------ End of Spawn Population Block ------------------
%% ------------------ Objective & Evaluation Block ------------------

function fitness = objectiveFunction(population, map)
    [m, n, ~] = size(population);
    fitness = zeros(m, 1);
    v0 = 1; % Speed in clear path
    v1 = v0 / 5; % Speed in obstacle (1/x times slower)

    % Define starting and ending points
    startPoint = [1, 1];
    endPoint = [size(map, 1), size(map, 2)];

    for i = 1:m
        % Get the path and add start and end points
        path = squeeze(population(i, :, :));
        path = [startPoint; path; endPoint]; % Insert start and end points

        totalTime = 0;

        % Adjust loop to include start and end points
        for j = 1:(size(path, 1) - 1)
            x0 = clampIndex(path(j, 1), size(map, 1));
            y0 = clampIndex(path(j, 2), size(map, 2));
            x1 = clampIndex(path(j+1, 1), size(map, 1));
            y1 = clampIndex(path(j+1, 2), size(map, 2));
            
            % Calculate time
            totalTime = totalTime + (sqrt((x1 - x0)^2 + (y1 - y0)^2) / v0) + (sqrt(collide(x0, y0, x1, y1, map)^2) / v1);
        end

        % Update fitness with total time for the path for fitness maximization
        fitness(i) = 1 / totalTime;
    end
end

function collisionLength = collide(x1, y1, x2, y2, map)
    dx = x2 - x1;
    dy = y2 - y1;
    n = max(abs(dx), abs(dy));
    x = round(linspace(x1, x2, n+1));
    y = round(linspace(y1, y2, n+1));
    
    collisionPoints = map(sub2ind(size(map), y, x)) == 0;
    dPoints = diff([false, collisionPoints, false]);
    startPoints = find(dPoints == 1);
    endPoints = find(dPoints == -1) - 1;
    
    collisionLength = round(sum(sqrt((x(endPoints) - x(startPoints)).^2 + (y(endPoints) - y(startPoints)).^2)));
end

function clampedIndex = clampIndex(index, maxIndex)
    clampedIndex = max(min(index, maxIndex), 1);
end
%% ------------------ End of Objective & Evaluation Block ------------------
%% ------------------ Selection Block ------------------

function offspring = selection(population, fitness, SelectionChoice)
    switch SelectionChoice
        case 1  % Roulette Wheel Selection
            offspring = rouletteWheelSelection(population, fitness);
        case 2  % Tournament Selection
            offspring = tournamentSelection(population, fitness);
        case 3  % Rank-Based Selection
            offspring = rankBasedSelection(population, fitness);
        otherwise
            error('Invalid Selection Choice');
    end
end

function selected = rouletteWheelSelection(population, fitness)
    m = size(population, 1);
    selectedSize = floor(0.8 * m);
    selected = zeros(selectedSize, size(population, 2), size(population, 3));
    cumulativeFitness = cumsum(fitness) / sum(fitness);

    for i = 1:selectedSize
        r = rand();
        idx = find(cumulativeFitness >= r, 1, 'first');
        selected(i, :, :) = population(idx, :, :);
    end
end

function selected = tournamentSelection(population, fitness)
    m = size(population, 1);
    tournamentSize = 2; % or any other small number
    selectedSize = floor(0.8 * m);
    selected = zeros(selectedSize, size(population, 2), size(population, 3));

    for i = 1:selectedSize
        contestants = randperm(m, tournamentSize);
        [~, bestIdx] = max(fitness(contestants));
        selected(i, :, :) = population(contestants(bestIdx), :, :);
    end
end

function selected = rankBasedSelection(population, fitness)
    m = size(population, 1);
    selectedSize = floor(0.8 * m);
    selected = zeros(selectedSize, size(population, 2), size(population, 3));

    % Rank the chromosomes: the best fitness gets the lowest rank number
    [~, sortedIdx] = sort(fitness, 'descend');  % Sort in descending order of fitness
    ranks = m:-1:1;  % Assign lower rank numbers to better fitness

    % Calculate selection probabilities based on inverse ranks
    totalRankSum = sum(ranks);
    probabilities = ranks / totalRankSum;

    % Assign probabilities to corresponding chromosomes
    rankBasedProbabilities = zeros(size(probabilities));
    for i = 1:m
        rankBasedProbabilities(sortedIdx(i)) = probabilities(i);
    end

    % Cumulative sum of probabilities for selection
    cumulativeProbabilities = cumsum(rankBasedProbabilities);

    for i = 1:selectedSize
        r = rand();
        idx = find(cumulativeProbabilities >= r, 1, 'first');
        selected(i, :, :) = population(idx, :, :);
    end
end
%% ------------------ End of Selection Block ------------------
%% ------------------ Crossover Block ------------------

function offspring = crossover(parents, crossoverRate, CrossOverChoice)
    switch CrossOverChoice
        case 1  % 2K-point Crossover
            offspring = vectorizedTwoPointCrossover(parents, crossoverRate);
        case 2  % Cycle Crossover
            offspring = cycleCrossover(parents, crossoverRate);
    end
end

function offspring = vectorizedTwoPointCrossover(parents, crossoverRate)
    offspring = parents;
    numParents = size(parents, 1);
    chromosomeLength = size(parents, 2);

    % Randomly decide which pairs will crossover
    crossoverPairs = rand(numParents / 2, 1) < crossoverRate;
    % Generate two random crossover points for each pair
    crossoverPoints = sort(randi([2, chromosomeLength - 1], numParents / 2, 2), 2);

    % Apply crossover for selected pairs
    for idx = find(crossoverPairs)'
        parent1Idx = 2 * idx - 1;
        parent2Idx = 2 * idx;

        startIdx = crossoverPoints(idx, 1);
        endIdx = crossoverPoints(idx, 2);

        % Swap segments between parents
        temp = offspring(parent1Idx, startIdx:endIdx, :);
        offspring(parent1Idx, startIdx:endIdx, :) = offspring(parent2Idx, startIdx:endIdx, :);
        offspring(parent2Idx, startIdx:endIdx, :) = temp;
    end
end

function offspring = cycleCrossover(parents, crossoverRate)
    [numChromosomes, chromosomeLength, ~] = size(parents);
    offspring = parents; % Initialize offspring with parents

    for i = 1:2:numChromosomes-1
        if rand <= crossoverRate
            % Perform cycle crossover on pairs of chromosomes
            p1 = squeeze(parents(i,:,:));
            p2 = squeeze(parents(i+1,:,:));

            [c1, c2] = cycleXover(p1, p2, chromosomeLength);
            offspring(i,:,:) = reshape(c1, [1, chromosomeLength, 2]);
            offspring(i+1,:,:) = reshape(c2, [1, chromosomeLength, 2]);
        end
    end
end

function [c1, c2] = cycleXover(p1, p2, chromosomeLength)
    c1 = p1;
    c2 = p2;

    % Start cycle from the first point
    cycle = false(1, chromosomeLength); 
    idx = 1;
    
    while ~cycle(idx)
        cycle(idx) = true;
        % Find the position in p1 matching the current point in p2
        idx = find(p1(:,1) == p2(idx,1) & p1(:,2) == p2(idx,2), 1, 'first');
    end

    % Swap the points in the cycle
    c1(cycle, :) = p2(cycle, :);
    c2(cycle, :) = p1(cycle, :);
end
%% ------------------ End of Crossover Block ------------------

%%------------------ Mutation Block ------------------
function offspring = mutation(offspring, mutationRate, MutationChoice)
    switch MutationChoice
        case 1  % Gaussian Mutation
            offspring = gaussianMutation(offspring, mutationRate);
        case 2  % Scramble Mutation
            offspring = scrambleMutation(offspring, mutationRate);
    end
end

function offspring = gaussianMutation(offspring, mutationRate)
    % Generate a random matrix of the same size as offspring
    mutationMask = rand(size(offspring)) < mutationRate;

    % Apply Gaussian mutation where the mask is true
    gaussianChanges = randn(size(offspring)) * mutationRate;
    offspring(mutationMask) = offspring(mutationMask) + gaussianChanges(mutationMask);

    % Round to nearest integer and ensure values are within bounds
    offspring = round(offspring);
    offspring = max(min(offspring, 500), 1);
end

function offspring = scrambleMutation(offspring, mutationRate)
    numIndividuals = size(offspring, 1);
    chromosomeLength = size(offspring, 2);

    for i = 1:numIndividuals
        if rand < mutationRate
            % Select two random indices to swap
            swapIndices = randperm(chromosomeLength, 2);

            % Swap the genes in both dimensions (x and y)
            temp = offspring(i, swapIndices(1), :);
            offspring(i, swapIndices(1), :) = offspring(i, swapIndices(2), :);
            offspring(i, swapIndices(2), :) = temp;
        end
    end

    % Ensure values are within bounds (if necessary)
    offspring = max(min(offspring, 500), 1);
end

%%------------------ End of Mutation Block ------------------


%%------------------ Review Block ------------------
function displayResults(population, fitness, map, start, finish)
    % Find the indices of the two best candidates
    [~, sortedIndices] = sort(fitness, 'descend');
    sortedPopulation = population(sortedIndices, :, :);

    % Display map and paths
    figure;
    imagesc(map); hold on;
    colormap(gray);
    axis equal;

    % Plot start and finish points
    plot(start(1), start(2), 'go'); % Start in green
    plot(finish(1), finish(2), 'ro'); % Finish in red

    for i = 1:2 % Loop over the two best candidates
        candidate = sortedPopulation(i, :, :); % Access the sorted population
        fitnessScore = fitness(sortedIndices(i)); % Access the sorted fitness

        % Plot path
        plotPath(squeeze(candidate));

        % Display points list and fitness
        disp(['Candidate ' num2str(i)]);
        disp('Path Points:');
        disp(squeeze(candidate));
        disp(['Fitness: ' num2str(fitnessScore)]);
        disp('-----------------------');
    end
    hold off;
end


function plotPath(path)
    path = squeeze(path);
    plot(path(:, 1), path(:, 2), 'LineWidth', 2);
end

function plotRequiredPathA(map, bestSolution, start, finish)
    % Reshape the bestSolution for plotting
    path = reshape(bestSolution, [], 2);
    % Scale the path according to the map size
    path = [start; path; finish]
    % Display the map
    imshow(map);
    hold on;
    % Plot the path
    plot(path(:,1), path(:,2), 'r-', 'LineWidth', 2);
    % Plot start and finish points
    plot(start(2), start(1), 'go', 'MarkerFaceColor', 'g'); % start in green
    plot(finish(2), finish(1), 'ro', 'MarkerFaceColor', 'r'); % finish in red

    hold off;
    title('Best Path Found by GA');
    xlabel('X Coordinate');
    ylabel('Y Coordinate');
end
%%------------------ End of Review Block ------------------