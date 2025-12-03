#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include <float.h>

int main(int argc, char *argv[]) {
    int p, my_rank;
    MPI_Status status;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &p);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    int n;                          // Total vector size
    double *X = NULL;               // Full vector X (only on rank 0)
    double *delta = NULL;           // Full vector delta (only on rank 0)
    
    double *local_X;                // Local part of X for each processor
    double *local_delta;            // Local part of delta for each processor
    
    int local_n;                    // Size of the local part
    int start_index;                // Starting index of the local part in X

    int choice = 0;

    do {
        // MENU AND DATA INPUT (RANK 0)

        if (my_rank == 0) {
            printf("\n--- OPTION MENU ---\n");
            printf("1. Continue Calculation\n");
            printf("2. Exit\n");
            printf("Selection: ");
            scanf("%d", &choice);

            // "Broadcast" the choice to workers
            for (int i = 1; i < p; i++) {
                MPI_Send(&choice, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
            }

            if (choice == 2) {
                break;
            }

            // Read input n
            printf("Enter the number of elements (n): ");
            scanf("%d", &n);

            // "Broadcast" n to workers
            for (int i = 1; i < p; i++) {
                MPI_Send(&n, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
            }

            // Create and populate the initial vector X
            X = (double*)malloc(n * sizeof(double));
            delta = (double*)malloc(n * sizeof(double)); // For the final delta vector
            if (X == NULL || delta == NULL) {
                printf("Memory allocation error on rank 0\n");
                MPI_Abort(MPI_COMM_WORLD, 1);
            }

            printf("Enter the %d elements of vector X:\n", n);
            for (int i = 0; i < n; i++) {
                // Using double for higher precision
                scanf("%lf", &X[i]);
            }

        } else {
            // Workers: Receive the choice from rank 0
            MPI_Recv(&choice, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);

            if (choice == 2) {
                break; // Exit the do-while loop
            }

            // Workers: Receive n from rank 0
            MPI_Recv(&n, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
        }

        if (choice == 2) {
            break; // Ensure exit for all
        }

        // DISTRIBUTION CALCULATION (n % p != 0)
        // Each processor calculates its own portion

        int base_chunk = n / p;
        int remainder = n % p;

        // The first 'remainder' processors take base_chunk + 1
        // The rest take base_chunk
        if (my_rank < remainder) {
            local_n = base_chunk + 1;
            start_index = my_rank * local_n;
        } else {
            local_n = base_chunk;
            start_index = (my_rank * base_chunk) + remainder;
        }
        
        // Memory allocation for local data
        local_X = (double*)malloc(local_n * sizeof(double));
        local_delta = (double*)malloc(local_n * sizeof(double));
        if (local_X == NULL || local_delta == NULL) {
            printf("Memory allocation error on rank %d\n", my_rank);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        // "SCATTER" (using Send/Recv)
        // Rank 0 sends parts to workers

        if (my_rank == 0) {
            // Rank 0 copies its own part
            for (int i = 0; i < local_n; i++) {
                local_X[i] = X[start_index + i];
            }

            // Rank 0 sends parts to others
            int current_start_index = local_n; // Start after our own part
            for (int i = 1; i < p; i++) {
                // Calculate size for worker 'i'
                int worker_local_n = (i < remainder) ? (base_chunk + 1) : (base_chunk);
                
                // Send the correct part of X
                MPI_Send(&X[current_start_index], worker_local_n, MPI_DOUBLE, i, 1, MPI_COMM_WORLD);
                current_start_index += worker_local_n;
            }
        } else {
            // Workers receive their own part
            MPI_Recv(local_X, local_n, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, &status);
        }

        // CALCULATION a) MEAN VALUE & b) MAXIMUM

        // a) Local Calculation (local_sum, local_max)
        double local_sum = 0.0;
        double local_max = DBL_MIN; // Initialize with the minimum possible double value

        for (int i = 0; i < local_n; i++) {
            local_sum += local_X[i];
            if (local_X[i] > local_max) {
                local_max = local_X[i];
            }
        }

        // b) "Reduce" (using Send/Recv) for m and mu
        double global_sum, global_max, mu, m;

        if (my_rank == 0) {
            global_sum = local_sum;
            global_max = local_max;

            // Rank 0 collects local sums and local maxes
            for (int i = 1; i < p; i++) {
                double received_sum, received_max;
                MPI_Recv(&received_sum, 1, MPI_DOUBLE, i, 2, MPI_COMM_WORLD, &status);
                MPI_Recv(&received_max, 1, MPI_DOUBLE, i, 3, MPI_COMM_WORLD, &status);
                
                global_sum += received_sum;
                if (received_max > global_max) {
                    global_max = received_max;
                }
            }

            // c) Final calculation of mu and m
            mu = global_sum / (double)n; // a) Mean value
            m = global_max;              // b) Max value

            // d) "Broadcast" (using Send/Recv) of mu and m to workers
            for (int i = 1; i < p; i++) {
                MPI_Send(&mu, 1, MPI_DOUBLE, i, 4, MPI_COMM_WORLD);
                MPI_Send(&m, 1, MPI_DOUBLE, i, 5, MPI_COMM_WORLD);
            }

        } else {
            // Workers send their local results to 0
            MPI_Send(&local_sum, 1, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD);
            MPI_Send(&local_max, 1, MPI_DOUBLE, 0, 3, MPI_COMM_WORLD);

            // Workers receive final mu and m from 0
            MPI_Recv(&mu, 1, MPI_DOUBLE, 0, 4, MPI_COMM_WORLD, &status);
            MPI_Recv(&m, 1, MPI_DOUBLE, 0, 5, MPI_COMM_WORLD, &status);
        }

        // CALCULATION c) VARIANCE & d) DELTA VECTOR

        // (a) Local calculation of variance sum and local_delta
        double local_var_sum = 0.0;
        for (int i = 0; i < local_n; i++) {
            // Calculate local sum for variance
            local_var_sum += (local_X[i] - mu) * (local_X[i] - mu);
            
            // Calculate local element for delta
            local_delta[i] = (local_X[i] - m) * (local_X[i] - m);
        }

        // b) "Reduce" (for var) & "Gather" (for delta) using Send/Recv

        double var; // Final variance

        if (my_rank == 0) {
            double global_var_sum = local_var_sum;

            // Copy rank 0's local delta to the final vector
            for (int i = 0; i < local_n; i++) {
                delta[start_index + i] = local_delta[i];
            }

            // Collect variance sums AND delta parts from workers
            int current_start_index = local_n;
            for (int i = 1; i < p; i++) {
                double received_var_sum;
                MPI_Recv(&received_var_sum, 1, MPI_DOUBLE, i, 6, MPI_COMM_WORLD, &status);
                global_var_sum += received_var_sum;

                // Calculate size and position for worker 'i'
                int worker_local_n = (i < remainder) ? (base_chunk + 1) : (base_chunk);
                
                // Receive delta part and place it in the correct position
                MPI_Recv(&delta[current_start_index], worker_local_n, MPI_DOUBLE, i, 7, MPI_COMM_WORLD, &status);
                current_start_index += worker_local_n;
            }

            // c) Final calculation of var
            var = global_var_sum / (double)n;

            // PRINT RESULTS (RANK 0)
            
            printf("\n--- CALCULATION RESULTS ---\n");
            printf("(a) Mean Value (mu): \t\t%.2f\n", mu);
            printf("(b) Max Value (m): \t\t%.2f\n", m);
            printf("(c) Variance (var): \t\t%.2f\n", var);
            
            printf("(d) New Vector (delta):\n[ ");
            for (int i = 0; i < n; i++) {
                printf("%.2f ", delta[i]);
            }
            printf("]\n");

            // Free memory for global vectors
            free(X);
            free(delta);

        } else {
            // Workers send variance sum to 0
            MPI_Send(&local_var_sum, 1, MPI_DOUBLE, 0, 6, MPI_COMM_WORLD);
            
            // Workers send their local delta part to 0
            MPI_Send(local_delta, local_n, MPI_DOUBLE, 0, 7, MPI_COMM_WORLD);
        }

        // Free memory for local vectors on ALL processors
        free(local_X);
        free(local_delta);

    } while (choice == 1); // The loop is controlled by the choice

    MPI_Finalize();
    return 0;
}
