#include <iostream>
#include <vector>
#include <random>
#include <algorithm> // Required for std::sort
#include <limits> // Required for input validation
#include <fstream>
#include <chrono> // Required for timing the simulation

// In this project, C++ code will simulate number of blackjack games using book moves to validate the true odds of winning blackjack. 
// The program will simulate a number of games and output the results to the console. 

struct hand
{
    std::vector<int> cards;
    int score = 0;
    bool isBust = false;
    bool isSoft = false;
    //double bet = 0;
    hand()
    {
        cards.reserve(10);
    }
    void addCard(int card)
    {
        cards.push_back(card);
    }
    void clear()
    {
        cards.clear();
        score = 0;
        isBust = false;
		isSoft = false;
    }
};

class game
{
private:

    const std::vector<std::string> cards = { "A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K" };
    hand playerCards;
    hand dealerCards;

    bool isRunning = true;
    bool noTen = false;
    bool noAce = false;

    // (Mersenne Twister) is a really fast prng
    std::mt19937 rng;
    std::uniform_int_distribution<int> random{ 1, 13 };
    std::uniform_int_distribution<int> random19{ 1, 9 };
    std::uniform_int_distribution<int> randomNoAce{ 2, 13 };
public:
    int winCount = 0;
    int loseCount = 0;
    int tieCount = 0;
    int handsPlayed = 0;

    game()
    {
        std::random_device rd;
        rng.seed(rd());
    }
	void clearTallies()
	{
		winCount = 0;
		loseCount = 0;
		tieCount = 0;
		handsPlayed = 0;
	}
    int getValue(int cardValue) const
    {
        switch (cardValue)
        {
        case 1: return 11; break; // we will default the value of an Ace to 11
        case 11: return 10; break;
        case 12: return 10; break;
        case 13: return 10; break;
        default:
            return cardValue;
        }
    }
    int sumCards(hand& heldCards)
    {
        // this function will instead check all of the values in the std::vector of cards the dealer or player has
        int score = 0;
        int ace_count = 0;

        for (int card : heldCards.cards)
        {
            if (card == 1)
            {
                ace_count++;
            }
            score += getValue(card);
        }

        // if the score is greater than 21 and there are aces, we will subtract 10 from the score for each ace until the score is less than or equal to 21 or there are no more aces
        while (score > 21 && ace_count > 0)
        {
            score -= 10;
            ace_count--;
        }
		
		heldCards.isSoft = (ace_count > 0); // if there are still aces left, the hand is soft
        return score;
    }
    int checkWin() const
    {
        // player win = 1
        // dealer win = 2
        // draw = 0
        if (playerCards.score > dealerCards.score && playerCards.score <= 21) return 1;
        else if (dealerCards.score > playerCards.score && dealerCards.score <= 21) return 2;
        return 0;
    }

    // overloaded function for when the player has split their hand and we need to check the win condition for each hand
    int checkWin(hand playerHand) const
    {
        // player win = 1
        // player lose = 2
        // draw = 0
        //std::cout << "Player Score: " << playerHand.score << "\n";
        if (dealerCards.score > 21)
        {
            return 1; // player wins if the dealer busts
        }
        else if (playerHand.score > dealerCards.score) return 1;
        else if (dealerCards.score > playerHand.score) return 2;
        return 0;
    }
    void dealerPlay(bool withCheckWin = true)
    {
        while (dealerCards.score <= 16)
        {
            if (noTen)
            {
                //std::cout << "Dealer is drawing a card that is not a 10.\n";
                dealerCards.addCard(random19(rng));
            }
            else if (noAce)
            {
                //std::cout << "Dealer is drawing a card that is not an Ace.\n";
                dealerCards.addCard(randomNoAce(rng));
            }
            else
            {
                dealerCards.addCard(random(rng));
            }
            dealerCards.score = sumCards(dealerCards);

			// ------------- DEBUGGING OUTPUT -------------
            //std::cout << "Dealer is showing:\n";
            //for (size_t i = 0; i < dealerCards.cards.size(); i++)
            //{
            //    std::cout << cards[dealerCards.cards[i] - 1] << " ";
            //}
            //std::cout << "\n";
            //std::cout << dealerCards.score << "\n";
            //std::cout << "**********************************\n";
            // ------------- DEBUGGING OUTPUT -------------
        }
        if (withCheckWin)
        {
            if (dealerCards.score > 21)
            {
                winCount++;
                dealerCards.isBust = true;
            }
            else
            {
                switch (checkWin())
                {
                case 1:
                    winCount++;
                    break;
                case 2:
                    loseCount++;
                    break;
                default:
                    tieCount++;
                    break;
                }
            }
        }
    }
    void initialize()
    {
        playerCards.clear();
        dealerCards.clear();
		handsPlayed++;

        const int dealerCard = random(rng);

        dealerCards.addCard(dealerCard);
        dealerCards.score += getValue(dealerCard);

        // give the player two cards
        playerCards.addCard(random(rng));
        playerCards.addCard(random(rng));
		playerCards.score = sumCards(playerCards);

        // check when the dealer has an A or 10 at the beginning of the game
        if (dealerCard == 1 || dealerCard >= 10)
        {
            // rng to test if 10
            const int test = random(rng);
            const bool dealerHasBlackjack = ((test >= 10 && dealerCard == 1) || (test == 1 && dealerCard >= 10));
			const bool playerHasBlackjack = (playerCards.score == 21);

            //std::cout << "Test number: " << test << "\n";
            if (dealerHasBlackjack)
            {
                if (playerHasBlackjack)
                {
					// both player and dealer have blackjack and it is a tie
                    tieCount++;
                }
                else
                {
                    // dealer has a blackjack
                    loseCount++;
                }
				isRunning = false;
            }
            else
            {
                if (playerHasBlackjack)
                {
                    winCount++;
                    isRunning = false;
                }
                if (dealerCard >= 10)
                {
                    noAce = true;
                }
                else
                {
                    noTen = true;
                }
            }
        }
    }
    int playerDecision(bool splittable, bool doubleDownable)
    {
        // this function will return the player's decision based on the cards they have and the dealer's up card
        // we will use the basic strategy chart to determine the player's decision
        // we will return 1 for hit, 2 for stand, 3 for double down, and 4 for split

        // https://www.vegas-aces.com/articles/blackjack-strategy-guides/

		// Check for player score, dealer up card, and whether the hand is soft or hard
		int playerScore = sumCards(playerCards); // the sumCards function updates whether the hand is soft or hard, so we can use that to determine the player's decision
		int dealerUpCard = dealerCards.cards[0];

        if (playerCards.isSoft)
        {
			// the only splittable soft hand is A,A, so we will handle that in the splittable section
            if (splittable)
            {
				// we always split A,A
				return 4; // split
            }
            else // everything else
            {
                if (playerScore >= 19) return 2;
				else if (playerScore == 18)
				{
                    if (dealerUpCard >= 3 && dealerUpCard <= 6)
                    {
                        if (doubleDownable) return 3; // double down
						else return 1; // hit
                    }
					else if (dealerUpCard == 2 || dealerUpCard == 7 || dealerUpCard == 8) return 2; // stand
					else return 1; // hit
				}
				else if (playerScore == 17)
				{
					if (dealerUpCard >= 3 && dealerUpCard <= 6)
                    {
                        if (doubleDownable) return 3; // double down
                        else return 1; // hit
                    }
					else return 1; // hit
				}
				else if (playerScore == 16 || playerScore == 15)
				{
					if (dealerUpCard >= 4 && dealerUpCard <= 6)
                    {
                        if (doubleDownable) return 3; // double down
                        else return 1; // hit
                    }
					else return 1; // hit
				}
				else if (playerScore == 14 || playerScore == 13)
				{
					if (dealerUpCard >= 5 && dealerUpCard <= 6)
                    {
                        if (doubleDownable) return 3; // double down
                        else return 1; // hit
                    }
					else return 1; // hit
				}
            }
        }
		else // hard hand
        {
            if (!splittable)
            {
                if (playerScore >= 17) return 2; // stand
                else if (playerScore >= 13 && playerScore <= 16)
                {
                    if (dealerUpCard >= 2 && dealerUpCard <= 6) return 2; // stand
                    else return 1; // hit
                }
                else if (playerScore == 12)
                {
                    if (dealerUpCard >= 4 && dealerUpCard <= 6) return 2; // stand
                    else return 1; // hit
                }
                else if (playerScore == 11)
                {
                    if (dealerUpCard == 1) return 1; // hit
                    else
                    {
                        if (doubleDownable) return 3; // double down
                        else return 1; // hit
                    }
                }
                else if (playerScore == 10)
                {
                    if (dealerUpCard >= 2 && dealerUpCard <= 9)
                    {
                        if (doubleDownable) return 3; // double down
                        else return 1; // hit
                    }
                    else return 1;
                }
                else if (playerScore == 9)
                {
                    if (dealerUpCard >= 3 && dealerUpCard <= 6)
                    {
                        if (doubleDownable) return 3; // double down
                        else return 1; // hit
                    }
					else return 1; // hit
                }
                else // player score is less than or equal to 8
                {
                    return 1; // hit
                }
            }
            else
            {
				// since both cards are the same, we can just check the value of one of the cards
                int cardValue = playerCards.cards[0];
                switch (cardValue)
                {
                case 2:
					if (dealerUpCard >= 2 && dealerUpCard <= 7) return 4; // split
					else return 1; // hit
                case 3:
                    if (dealerUpCard >= 2 && dealerUpCard <= 7) return 4; // split
                    else return 1; // hit
				case 4:
					if (dealerUpCard >= 5 && dealerUpCard <= 6) return 4; // split
					else return 1; // hit
				case 5:
					if (dealerUpCard >= 2 && dealerUpCard <= 9)
                    {
                        if (doubleDownable) return 3; // double down
                        else return 1; // hit
                    }
					else return 1; // hit
				case 6:
					if (dealerUpCard >= 2 && dealerUpCard <= 6) return 4; // split
					else return 1; // hit
				case 7:
					if (dealerUpCard >= 2 && dealerUpCard <= 7) return 4; // split
					else return 1; // hit
				case 8:
					return 4; // always split 8s
				case 9:
					if (dealerUpCard >= 2 && dealerUpCard <= 6) return 4; // split
					else if (dealerUpCard == 8 || dealerUpCard == 9) return 4; // split
					else return 2; // stand
				default:
					// the only other splittable hard hand is 10s and face cards which we will always stand on
					return 2; // stand
                }
            }
        }
    }
    void handleSplit()
    {
        // this function will handle the split functionality
        // we will need to create a new hand for the player and give them a new card for each hand
        std::vector<hand> splitHands;
        splitHands.reserve(5);
		handsPlayed++; // increment the hands played counter since we are splitting the hand

        playerCards.cards.pop_back(); // remove the second card from the original hand
        splitHands.push_back(playerCards); // now add the hand with the first card to the splitHands vector twice
        splitHands.push_back(playerCards);
        splitHands[0].addCard(random(rng)); // give the first hand a new card
        splitHands[1].addCard(random(rng));
        splitHands[0].score = sumCards(splitHands[0]); // update the score for the first hand
        splitHands[1].score = sumCards(splitHands[1]); // update the score for the second hand

        for (size_t i = 0; i < splitHands.size(); i++)
        {
            playerCards = splitHands[i];
            playerCards.score = sumCards(playerCards);

            //std::cout << "**********************************\n";
            //std::cout << "Hand " << i + 1 << ":\n";
            //std::cout << "Player is showing: ";
            //for (size_t j = 0; j < playerCards.cards.size(); j++)
            //{
            //    std::cout << cards[playerCards.cards[j] - 1] << " ";
            //}
            //std::cout << "(" << playerCards.score << ") ";
            //std::cout << "vs. the dealer's up card " << cards[dealerCards.cards[0] - 1] << "\n";

            // don't need to check if player has 21 right away

            bool handRunning = true;
            while (handRunning)
            {
                // if the player only has two cards and they are both the same card, they have the option to split
                bool splittable = (playerCards.cards.size() == 2 && playerCards.cards[0] == playerCards.cards[1]);
				bool doubleDownable = (playerCards.cards.size() == 2);

				// We will have to set the rules for what the player will hit, stand, double down, or split based on the cards they have and the dealer's up card
				int playerMove = playerDecision(splittable, doubleDownable);
                switch (playerMove)
                {
                case 1:
                    // Hit
                    //std::cout << "Player is showing: ";
                    //for (size_t i = 0; i < playerCards.cards.size(); i++)
                    //{
                    //    std::cout << cards[playerCards.cards[i] - 1] << " ";
                    //}
                    //std::cout << "(" << playerCards.score << ") ";
                    //std::cout << "vs. the dealer's up card " << cards[dealerCards.cards[0] - 1] << " and is hitting.\n";

                    playerCards.addCard(random(rng));
                    playerCards.score = sumCards(playerCards);
                    //std::cout << "(" << playerCards.score << ") \n";

                    if (playerCards.score > 21)
                    {
						loseCount++;
                        playerCards.isBust = true;
                        handRunning = false;
                    }
                    else if (playerCards.score == 21)
                    {
                        // We'll just move onto the next hand if the player has 21
                        handRunning = false;
                    }
					// else we will just continue the while loop and let the player hit again
                    break;
                case 2:
                    // Stand
                    handRunning = false;
                    break;
                case 3:
                    // Double Down
                    //std::cout << "Player is showing: ";
                    //for (size_t i = 0; i < playerCards.cards.size(); i++)
                    //{
                    //    std::cout << cards[playerCards.cards[i] - 1] << " ";
                    //}
                    //std::cout << "(" << playerCards.score << ") ";
                    //std::cout << "vs. the dealer's up card " << cards[dealerCards.cards[0] - 1] << " and is doubling down.\n";

                    playerCards.addCard(random(rng));
                    playerCards.score = sumCards(playerCards);
                    //std::cout << "(" << playerCards.score << ") \n";

                    if (playerCards.score > 21)
                    {
                        loseCount++;
                        playerCards.isBust = true;
                    }
                    handRunning = false;
                    break;
                case 4:
                    // Split
                    if (splittable)
                    {
						handsPlayed++; // increment the hands played counter since we are splitting the hand
                        playerCards.cards.pop_back(); // remove the second card from the current hand
                        playerCards.score = sumCards(playerCards); // update the score for the current hand

                        hand newHand = playerCards; // create a new hand with the current hand's cards
                        newHand.addCard(random(rng)); // give the new hand a new card
                        newHand.score = sumCards(newHand); // update the score for the new hand

                        splitHands.push_back(newHand); // add the new hand to the splitHands vector

                        // now we want the while loop to continue with the current hand, and then move onto the next hand in the splitHands vector
                        playerCards.addCard(random(rng)); // give the current hand a new card
                        playerCards.score = sumCards(playerCards);

                        if (playerCards.score == 21)
                        {
                            splitHands[i] = playerCards;
                            handRunning = false;
                        }
						// else we will just continue the while loop and let the player decide again
                    }
                    break;
                default:
                    break;
                }
                splitHands[i] = playerCards; // update the splitHands vector with the current hand
            }
        }
        // Once the for loop is done, we will have to play the dealer's hand and then check the win condition for each hand in the splitHands vector
		dealerPlay(false); // simply play the dealer's hand without checking the win condition yet, since we will do that for each hand in the splitHands vector

        for (size_t i = 0; i < splitHands.size(); i++)
        {
            const hand& currentHand = splitHands[i];
            if (currentHand.isBust)
            {
				//std::cout << "Hand " << i + 1 << " is bust. You lose.\n";
				loseCount++;
            }
            else
            {
                switch (checkWin(currentHand))
                {
                case 1:
					//std::cout << "Hand " << i + 1 << " wins!\n";
					winCount++;
                    break;
                case 2:
					//std::cout << "Hand " << i + 1 << " loses.\n";
					loseCount++;
                    break;
                default:
					//std::cout << "Hand " << i + 1 << " is a tie.\n";
					tieCount++;
                    break;
                }
            }
        }
    }
    void play()
    {
        while (isRunning)
        {
            bool splittable = (playerCards.cards.size() == 2 && playerCards.cards[0] == playerCards.cards[1]);
			bool doubleDownable = (playerCards.cards.size() == 2);

            // We will have to set the rules for what the player will hit, stand, double down, or split based on the cards they have and the dealer's up card
            int playerMove = playerDecision(splittable, doubleDownable);

            switch (playerMove)
            {
            case 1:
                // Hit
    //            std::cout << "Player is showing: ";
				//for (size_t i = 0; i < playerCards.cards.size(); i++)
				//{
				//	std::cout << cards[playerCards.cards[i] - 1] << " ";
				//}
				//std::cout << "(" << playerCards.score << ") ";
    //            std::cout << "vs. the dealer's up card " << cards[dealerCards.cards[0] - 1] << " and is hitting.\n";

                playerCards.addCard(random(rng));
                playerCards.score = sumCards(playerCards);
                //std::cout << "(" << playerCards.score << ") \n";

                if (playerCards.score > 21)
                {
					loseCount++;
                    playerCards.isBust = true;
                    isRunning = false;
                }
                else if (playerCards.score == 21)
                {
                    dealerPlay();
                    isRunning = false;
                }
                // else we will just continue the while loop and let the player hit again
                break;
            case 2:
                // Stand
                //std::cout << "Player is showing: ";
                //for (size_t i = 0; i < playerCards.cards.size(); i++)
                //{
                //    std::cout << cards[playerCards.cards[i] - 1] << " ";
                //}
                //std::cout << "(" << playerCards.score << ") ";
                //std::cout << "vs. the dealer's up card " << cards[dealerCards.cards[0] - 1] << " and is standing.\n";

                dealerPlay();
                isRunning = false;
                break;
            case 3:
                // Double Down
                //std::cout << "Player is showing: ";
                //for (size_t i = 0; i < playerCards.cards.size(); i++)
                //{
                //    std::cout << cards[playerCards.cards[i] - 1] << " ";
                //}
                //std::cout << "(" << playerCards.score << ") ";
                //std::cout << "vs. the dealer's up card " << cards[dealerCards.cards[0] - 1] << " and is doubling down.\n";

                playerCards.addCard(random(rng));
                playerCards.score = sumCards(playerCards);
                //std::cout << "(" << playerCards.score << ") \n";

                if (playerCards.score > 21)
                {
                    loseCount++;
                    playerCards.isBust = true;
                }
                else
                {
                    dealerPlay();
                }
                isRunning = false;
                break;
            case 4:
                // Split
                //std::cout << "Player is showing: ";
                //for (size_t i = 0; i < playerCards.cards.size(); i++)
                //{
                //    std::cout << cards[playerCards.cards[i] - 1] << " ";
                //}
                //std::cout << "(" << playerCards.score << ") ";
                //std::cout << "vs. the dealer's up card " << cards[dealerCards.cards[0] - 1] << " and is splitting.\n";

                if (splittable)
                {
                    handleSplit();
                    isRunning = false;
                }
                break;
            default:
                break;
            }
        }
        isRunning = true;
        noTen = false;
        noAce = false;
    }
};

void simulate(const int numGames, const int numTests)
{
	// TODO: Add code to simulate a number of blackjack games and output the results to a csv file
    std::ofstream file("blackjack_results.csv");
    file << "Win Prob,Loss Prob,Tie Prob\n";
    if (!file.is_open())
    {
        std::cerr << "Error opening file!" << std::endl;
    }

    for (int test = 0; test < numTests; test++)
    {
        game currentGame;
        for (int i = 0; i < numGames; i++)
        {
            currentGame.initialize();
            currentGame.play();
        }
		file << static_cast<double>(currentGame.winCount) / currentGame.handsPlayed << ","
			 << static_cast<double>(currentGame.loseCount) / currentGame.handsPlayed << ","
			 << static_cast<double>(currentGame.tieCount) / currentGame.handsPlayed << "\n";
        //std::cout << currentGame.handsPlayed << "\n";
    }
	std::cout << "Simulation complete. Results saved to blackjack_results.csv\n";
    file.close();
}

int main()
{
	// I want to measure runtime of the simulation, so I will use the chrono library to measure the time it takes to run the simulation
	auto start = std::chrono::high_resolution_clock::now();

	int numGames = 1000; // Number of games to simulate per test
	int numTests = 10000; // Number of tests to run
    simulate(numGames, numTests);

	auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration<double>(end - start);
	std::cout << "Simulation took " << duration.count() << " seconds to run.\n";
    return 0;
}
